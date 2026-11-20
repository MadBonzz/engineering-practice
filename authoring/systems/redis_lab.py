"""A small RESP2 server with a deterministic byte-stream replay and real TCP adapter."""

from textwrap import dedent


def module(source):
    return dedent(source).strip() + "\n"


MODULES = {
    "protocol.py": module(r'''
        """Decode one RESP2 request from the beginning of a TCP receive buffer."""

        class NeedMore(Exception):
            """Internal signal for an incomplete but still possible frame."""

        class ProtocolError(ValueError):
            """The peer sent a malformed or out-of-range frame."""

        def header(data, start, marker, maximum):
            """Read MARKER<decimal>CRLF. Return (nonnegative length, next offset)."""
            if start >= len(data):
                raise NeedMore
            if data[start:start + 1] != marker:
                raise ProtocolError("unexpected type marker")
            end = data.find(b"\r\n", start + 1)
            if end == -1:
                if len(data) - start > 11:
                    raise ProtocolError("header too long")
                raise NeedMore
            digits = data[start + 1:end]
            if not 1 <= len(digits) <= 9 or not digits.isdigit():
                raise ProtocolError("bad length")
            length = int(digits)
            if length > maximum:
                raise ProtocolError("length exceeds lab limit")
            return length, end + 2

        def parse_request(data):
            """Return (list of bytes arguments, consumed byte count) for the first complete frame, None for an incomplete frame, or raise ProtocolError for a malformed frame. Use header() with limits 1024 arguments and 65536 bytes per bulk string. Reject empty command arrays."""
            try:
                count, offset = header(data, 0, b"*", 1024)
                if count == 0:
                    raise ProtocolError("empty command array")
                args = []
                for _ in range(count):
                    size, offset = header(data, offset, b"$", 65536)
                    end = offset + size
                    if len(data) < end + 2:
                        raise NeedMore
                    if data[end:end + 2] != b"\r\n":
                        raise ProtocolError("missing bulk terminator")
                    args.append(data[offset:end])
                    offset = end + 2
                return args, offset
            except NeedMore:
                return None
    '''),
    "store.py": module('''
        """The shared database maps byte keys to (byte value, expiry_ms or None)."""

        def lookup(records, key, now_ms):
            """Return the live bytes value, or None for missing/expired keys. A deadline is exclusive: expire and remove the record when now_ms >= expiry_ms."""
            record = records.get(key)
            if record is None:
                return None
            value, deadline = record
            if deadline is not None and now_ms >= deadline:
                del records[key]
                return None
            return value

        def put(records, key, value, now_ms, ttl_ms=None):
            """Replace any old record, setting an absolute deadline of now_ms + ttl_ms. A missing TTL clears any prior expiry. The dispatcher validates TTLs."""
            deadline = None if ttl_ms is None else now_ms + ttl_ms
            records[key] = (value, deadline)

        def delete_many(records, keys, now_ms):
            """Delete keys and return the count of live entries removed. Missing, expired and repeated keys contribute zero after their first removal."""
            removed = 0
            for key in keys:
                if lookup(records, key, now_ms) is not None:
                    del records[key]
                    removed += 1
            return removed
    '''),
    "commands.py": module(r'''
        """Translate raw command arguments to a single reply on the wire."""
        from store import delete_many, lookup, put

        CRLF = b"\r\n"
        ARITY = b"-ERR wrong number of arguments\r\n"
        SYNTAX = b"-ERR syntax error\r\n"
        EXPIRY = b"-ERR invalid expire time\r\n"
        UNKNOWN = b"-ERR unknown command\r\n"

        def bulk(value):
            """Encode None as a null bulk string and bytes as a length-prefixed string."""
            if value is None:
                return b"$-1\r\n"
            return b"$" + str(len(value)).encode("ascii") + CRLF + value + CRLF

        def execute(records, args, now_ms):
            """Return RESP2 bytes for PING [message], SET key value [PX ms], GET key or DEL key [key ...]. Ignore ASCII case in the verb and PX. Validate before writing. See constants for the four command errors."""
            verb = args[0].upper()
            if verb == b"PING":
                if len(args) == 1:
                    return b"+PONG\r\n"
                return bulk(args[1]) if len(args) == 2 else ARITY
            if verb == b"GET":
                return bulk(lookup(records, args[1], now_ms)) if len(args) == 2 else ARITY
            if verb == b"DEL":
                if len(args) < 2:
                    return ARITY
                count = delete_many(records, args[1:], now_ms)
                return b":" + str(count).encode("ascii") + CRLF
            if verb == b"SET":
                if len(args) < 3:
                    return ARITY
                ttl = None
                if len(args) > 3:
                    if len(args) != 5 or args[3].upper() != b"PX":
                        return SYNTAX
                    text = args[4]
                    if not 1 <= len(text) <= 9 or not text.isdigit() or int(text) == 0:
                        return EXPIRY
                    ttl = int(text)
                put(records, args[1], args[2], now_ms, ttl)
                return b"+OK\r\n"
            return UNKNOWN
    '''),
    "session.py": module(r'''
        """Every connection owns one receive buffer; all connections share one database."""
        from commands import execute
        from protocol import ProtocolError, parse_request

        class Session:
            def __init__(self, records):
                self.records = records
                self.pending = b""
                self.closed = False

            def feed(self, chunk, now_ms):
                self.pending += chunk
                output = []
                offset = 0
                while offset < len(self.pending):
                    try:
                        parsed = parse_request(self.pending[offset:])
                    except ProtocolError:
                        output.append(b"-ERR protocol error\r\n")
                        self.pending = b""
                        self.closed = True
                        return b"".join(output)
                    if parsed is None:
                        break
                    args, used = parsed
                    output.append(execute(self.records, args, now_ms))
                    offset += used
                self.pending = self.pending[offset:]
                return b"".join(output)
    '''),
    "gateway.py": module('''
        """Replay deterministic byte chunks and per-connection disconnects."""
        from session import Session

        def handle(request):
            records = {}
            clients = {}
            replies = []
            for event in request["events"]:
                name = event["client"]
                if event.get("close"):
                    clients.pop(name, None)
                    replies.append("")
                    continue
                if name not in clients:
                    clients[name] = Session(records)
                client = clients[name]
                reply = client.feed(event["wire"].encode("latin1"), event["at"])
                replies.append(reply.decode("latin1"))
                if client.closed:
                    del clients[name]
            pending = {c: len(s.pending) for c, s in clients.items() if s.pending}
            return {"replies": replies, "pending": pending}
    '''),
    "server.py": module('''
        """Run python server.py --port 6380 in a terminal. Listens only on 127.0.0.1."""
        import argparse
        import asyncio
        import time

        from session import Session

        def connection_handler(records):
            async def handle(reader, writer):
                session = Session(records)
                try:
                    while not session.closed:
                        chunk = await reader.read(65536)
                        if not chunk:
                            break
                        now_ms = time.monotonic_ns() // 1_000_000
                        writer.write(session.feed(chunk, now_ms))
                        await writer.drain()
                except (ConnectionResetError, BrokenPipeError):
                    pass
                finally:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except (ConnectionResetError, BrokenPipeError):
                        pass
            return handle

        async def main(port):
            server = await asyncio.start_server(connection_handler({}), "127.0.0.1", port)
            async with server:
                print(f"Listening on 127.0.0.1:{server.sockets[0].getsockname()[1]}", flush=True)
                await server.serve_forever()

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, default=6380)
            try:
                asyncio.run(main(parser.parse_args().port))
            except KeyboardInterrupt:
                pass
    '''),
    "demo.py": module(r'''
        """With the server running, use python demo.py --port 6380 in a second terminal."""
        import argparse
        import socket

        def request(*args):
            frame = [b"*" + str(len(args)).encode() + b"\r\n"]
            for arg in args:
                frame += [b"$" + str(len(arg)).encode() + b"\r\n", arg, b"\r\n"]
            return b"".join(frame)

        def expect(stream, expected):
            result = stream.read(len(expected))
            if result != expected:
                raise AssertionError(f"Expected {expected!r}, received {result!r}")
            print(repr(result))

        def main(port):
            addr = ("127.0.0.1", port)
            with socket.create_connection(addr, timeout=3) as first, socket.create_connection(addr, timeout=3) as second:
                with first.makefile("rb") as first_input, second.makefile("rb") as second_input:
                    value = b"A\x00\r\n\xff"
                    packet = request(b"SET", b"lab:demo", value)
                    first.sendall(packet[:9])
                    second.sendall(request(b"PING"))
                    expect(second_input, b"+PONG\r\n")
                    first.sendall(packet[9:])
                    expect(first_input, b"+OK\r\n")
                    second.sendall(request(b"GET", b"lab:demo") + request(b"DEL", b"lab:demo"))
                    expect(second_input, b"$5\r\n" + value + b"\r\n:1\r\n")

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, default=6380)
            main(parser.parse_args().port)
    '''),
    "probes.py": module('''
        """JSON-friendly probes for the real byte-oriented module interfaces."""
        from commands import execute
        from protocol import parse_request, ProtocolError
        from store import delete_many, lookup, put

        def frame(wire):
            try:
                result = parse_request(wire.encode("latin1"))
            except ProtocolError:
                return {"error": "protocol"}
            if result is None:
                return None
            args, used = result
            return {"args": [a.decode("latin1") for a in args], "used": used}

        def store_trace(ops):
            records, results = {}, []
            for op, now, *args in ops:
                key = args[0].encode("latin1")
                if op == "put":
                    put(records, key, args[1].encode("latin1"), now, args[2])
                    results.append(None)
                elif op == "get":
                    value = lookup(records, key, now)
                    results.append(None if value is None else value.decode("latin1"))
                elif op == "del":
                    results.append(delete_many(records, [x.encode("latin1") for x in args], now))
            return results

        def command_trace(ops):
            records = {}
            replies = []
            for now, arguments in ops:
                wire = execute(records, [arg.encode("latin1") for arg in arguments], now)
                replies.append(wire.decode("latin1"))
            return replies
    '''),
}


def request(*arguments):
    pieces = [f"*{len(arguments)}\r\n"]
    for argument in arguments:
        pieces.extend([f"${len(argument.encode('latin1'))}\r\n", argument, "\r\n"])
    return "".join(pieces)


def recv(at, client, wire):
    return {"at": at, "client": client, "wire": wire}


def close(at, client):
    return {"at": at, "client": client, "close": True}


def trace(*events):
    return {"events": list(events)}


def output(*replies, pending=None):
    return {"replies": list(replies), "pending": pending or {}}


def check(part, title, fn, args, oracle):
    return dict(part=part, title=title, fn=fn, args=args, oracle=oracle)


def full(part, title, payload, answer):
    return check(part, title, "gateway.handle", [payload], answer)


ok, pong, missing = "+OK\r\n", "+PONG\r\n", "$-1\r\n"
empty, one = "$0\r\n\r\n", ":1\r\n"
get_k, ping = request("GET", "k"), request("PING")
set_px = request("SET", "k", "v", "PX", "5")
binary = "A\x00\r\nÿ"

EXAMPLES = [
    dict(input=trace(recv(10, "writer", set_px), recv(14, "reader", get_k), recv(15, "reader", get_k)),
         output=output(ok, "$1\r\nv\r\n", missing),
         why="The writer sets k at 10 ms with PX 5. A second connection sees v at 14 ms. At exactly 15 ms GET returns a RESP null bulk string."),
    dict(input=trace(recv(0, "A", "*1\r\n$4\r\nPI"), recv(1, "B", ping), recv(2, "A", "NG\r\n" + request("PING", "\x00ÿ") + "*2\r")),
         output=output("", pong, pong + "$2\r\n\x00ÿ\r\n", pending={"A": 3}),
         why="A's first PING is incomplete. B has a separate receive buffer and replies immediately. The last read finishes A's PING, echoes two binary bytes, and retains the three bytes of a third request.")
]
CHECKS = [
    check(1, "Consume one request and leave the following frame", "probes.frame", [ping + get_k],
          {"args": ["PING"], "used": 14}),
    check(1, "Truncated final terminator needs another byte", "probes.frame", [request("PING", binary)[:-1]], None),
    check(1, "Bulk length spans CRLF and non-UTF-8 data", "probes.frame", [request("PING", binary)],
          {"args": ["PING", binary], "used": 25}),
    check(1, "Malformed bulk terminator", "probes.frame", ["*1\r\n$1\r\nKxx"], {"error": "protocol"}),
    check(1, "Empty array violates the command grammar", "probes.frame", ["*0\r\n"], {"error": "protocol"}),
    check(2, "Lookup immediately before and at an expiry", "probes.store_trace", [[
        ["put", 20, "a", "v", 3], ["get", 22, "a"], ["get", 23, "a"], ["del", 23, "a"]]], [None, "v", None, 0]),
    check(2, "Empty value and repeated key count exactly once", "probes.store_trace", [[
        ["put", 0, "", "", None], ["del", 0, "missing", "", ""], ["get", 0, ""]]], [None, 1, None]),
    check(2, "A plain SET clears a previous deadline", "probes.store_trace", [[
        ["put", 0, "k", "old", 2], ["put", 1, "k", "v", None], ["get", 2, "k"]]], [None, None, "v"]),
    check(3, "Basic PING and binary echo", "probes.command_trace", [[[0, ["pinG"]], [0, ["PING", "\x00"]]]],
          [pong, "$1\r\n\x00\r\n"]),
    check(3, "An invalid PX leaves a prior value intact", "probes.command_trace", [[[0, ["SET", "k", "old"]], [2, ["SET", "k", "new", "pX", "-1"]], [3, ["get", "k"]]]],
          [ok, "-ERR invalid expire time\r\n", "$3\r\nold\r\n"]),
    check(3, "Arity, syntax and unsupported commands use different errors", "probes.command_trace",
          [[[0, ["DEL"]], [0, ["SET", "k", "v", "PX"]], [0, ["INCR", "k"]]]],
          ["-ERR wrong number of arguments\r\n", "-ERR syntax error\r\n", "-ERR unknown command\r\n"]),
    full(1, "Interleave two partial commands in separate sessions",
         trace(recv(0,"a",ping[:5]), recv(1,"b",get_k[:12]), recv(2,"a",ping[5:]), recv(3,"b",get_k[12:])),
         output("", "", pong, missing)),
    full(1, "Disconnect discards only the incomplete command",
         trace(recv(0,"a",request("SET","k","old") + ping[:8]), close(1,"a"), recv(2,"a",get_k)),
         output(ok, "", "$3\r\nold\r\n")),
    full(1, "Valid prefix precedes a framing error and a fresh connection",
         trace(recv(0,"a",ping + "*1\r\n$1\r\nXzz" + ping), recv(1,"a",ping)),
         output(pong + "-ERR protocol error\r\n", pong)),
    full(2, "Write happens when the last byte arrives", trace(recv(10,"a",set_px[:-1]), recv(30,"b",get_k),
         recv(40,"a",set_px[-1:]), recv(44,"b",get_k), recv(45,"b",request("DEL","k"))),
         output("", missing, ok, "$1\r\nv\r\n", ":0\r\n")),
    full(2, "Replacement removes the old deadline and preserves empty bytes",
         trace(recv(0,"a",set_px), recv(4,"b",request("SET","k","")), recv(9,"a",get_k+request("DEL","k","k"))),
         output(ok, ok, empty + one)),
    full(3, "Pipeline recovers after a command error and preserves byte identity",
         trace(recv(0,"reader",request("SET","\xff",binary) + request("NOOP")+request("GET","\xff"))),
         output(ok + "-ERR unknown command\r\n" + "$5\r\n" + binary + "\r\n")),
    full(3, "Two clients receive no reply for partial frames and retain exact tails",
         trace(recv(0,"a",ping+get_k[:5]), recv(0,"b",get_k[:-1])),
         output(pong, "", pending={"a":5, "b":len(get_k)-1})),
]

LABS = [dict(
    slug="sys-mini-redis", title="Build a Redis-compatible TCP server",
    group="Database internals", level="Advanced", duration="2–4 hours",
    mission="Two clients share a keyspace. One sends half a request and pauses. The other pipelines a SET and a GET containing a newline and a zero byte. Implement the server that decodes the stream, applies commands, and sends each reply to the right socket. Then connect two real clients to the same code on localhost.",
    scope="Build the RESP2 request parser, in-memory byte store and command dispatcher. The project supplies the asyncio listener, connection buffering and a replay adapter. The server handles PING [message], SET key value [PX ms], GET key and DEL key [key ...]. Data lives for the lifetime of a single server process. The tests focus on stream fragmentation, command ordering, binary values and expiration.",
    inputs='Pass {"events": [...]} to gateway.handle. Each ordered event contains at (integer milliseconds), client (connection ID), and either wire (a string of bytes that arrived) or close: true. In JSON one Latin-1 code point represents one byte, so \\r\\n is the RESP delimiter and \\u00ff is byte FF. A client ID opens a connection on its first read and can reconnect after a close. Times never decrease.',
    outputs='Return {"replies": [...], "pending": {...}}. replies[i] contains all wire bytes generated by event i, in command order. An incomplete frame or disconnect emits an empty string. pending maps each still-connected client with buffered data to its unread byte count. A new replay starts with empty storage and connections.',
    constraints="Every request is a nonempty RESP2 array of up to 1024 bulk strings, each at most 65536 bytes. Use the supplied header() grammar. A command takes effect at the timestamp of the chunk that completes its frame. Preserve byte-valued keys and values, including empty strings. A malformed frame emits the protocol error, discards the remaining stream and closes that connection. A valid but unsupported or misused command emits the documented command error and processing continues.",
    localRun="From the extracted starter, run python server.py --port 6380. In a second terminal run python demo.py --port 6380. The demo opens two connections, fragments a SET and checks the exact binary reply from a pipelined GET and DEL. You can also send individual commands with redis-cli -h 127.0.0.1 -p 6380 PING. Stop the server with Ctrl+C.",
    source=[
        dict(title="Redis · RESP wire protocol",url="https://redis.io/docs/latest/develop/reference/protocol-spec/"),
        dict(title="Redis · SET and expiration",url="https://redis.io/docs/latest/commands/set/"),
        dict(title="Redis · PING",url="https://redis.io/docs/latest/commands/ping/"),
        dict(title="Redis · DEL",url="https://redis.io/docs/latest/commands/del/"),
    ],
    modules=MODULES,
    parts=[
        dict(title="Frame a TCP byte stream",file="protocol.py",funcs=["parse_request"],
             requirement="Use header() to consume an array length and one bulk-length header per argument. Read exactly the promised payload bytes and its two-byte CRLF. Return only the first complete command and its consumed length. On incomplete input return None, leaving buffering to Session. A malformed type, empty array, out-of-range length or invalid complete terminator raises ProtocolError.",
             hints=["A CRLF inside the bulk payload is ordinary data. Apply the delimiter check at start + size.", "A complete first frame can be followed by arbitrary bytes. Stop after its last terminator."]),
        dict(title="Own the clock and the keyspace",file="store.py",funcs=["lookup","put","delete_many"],
             requirement="Implement the shared map of byte keys to (value, absolute_deadline_or_None). Enforce expiry on read and delete. A plain overwrite clears the old deadline. DEL counts each live key removed, including keys containing empty byte values.",
             hints=["The byte string b'' is a live value. Reserve None to represent absence.", "Use the clock passed by the caller, including when it is exactly equal to the deadline."]),
        dict(title="Dispatch commands and serve real clients",file="commands.py",funcs=["bulk","execute"],
             requirement="Use ARITY for invalid PING/GET/DEL arity or SET missing key/value. SET with extra arguments accepts exactly PX and its duration; use SYNTAX for other extension shapes and EXPIRY for invalid duration (requires 1–9 ASCII digits and positive value). Use UNKNOWN for unsupported commands. Bare PING replies +PONG; PING with a message and GET return bulk strings. Successful SET replies +OK and DEL returns a RESP integer. Every response ends in CRLF. Start server.py and run demo.py from a second terminal.",
             hints=["A null GET encodes as $-1 followed by CRLF. An empty value encodes as $0 followed by two CRLFs.", "Validate the entire SET before writing. One command error must not swallow the next pipelined request."]),
    ],
    examples=EXAMPLES, checks=CHECKS,
)]
