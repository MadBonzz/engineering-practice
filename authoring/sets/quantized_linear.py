from struct import pack, unpack


def quantize(row):
    """Quantize one nonempty output channel using signed symmetric int8 values [-127,127]. Scale is max(abs(weight))/127, or 1.0 for an all-zero row. Round weight/scale using Python round (ties to even). Return [scale, integer_codes]."""
    peak = max(map(abs,row))
    scale = peak/127 if peak else 1.
    return [scale,[max(-127,min(127,round(v/scale))) for v in row]]


def project(rows, x, bias):
    """rows is a list of [scale, codes] channels. Compute output_i = bias_i + scale_i * dot(codes_i,x). Keep codes in their integer representation in storage. Return one number per channel."""
    return [b+s*sum(q*v for q,v in zip(c,x)) for (s,c),b in zip(rows,bias)]


def export(weights):
    """Serialize each row as a little-endian float32 scale followed immediately by its signed int8 codes, with no header, padding, or bias. Return lowercase hex so the wire image can be compared exactly. A reader knows the dimensions. Empty weights yield an empty string."""
    payload = bytearray()
    for row in weights:
        scale,codes = quantize(row)
        payload.extend(pack("<f",scale))
        payload.extend(pack(f"{len(codes)}b",*codes))
    return payload.hex()

# AUTHORING
TITLE = "Quantized linear inference"
GROUP = "Inference"
INTRO = "A model exporter and a CPU inference worker must agree on a compact weight format. Calibrate output channels independently, run the layer, and serialize exactly what it stores."
PARTS = ["quantize","project","export"]
LABELS = ["Calibrate one channel","Run the quantized projection","Export the packed weights"]
HINTS = ["Try positive-only and all-zero rows.|Scale is per channel.", "Bias is in the original output units.|Different rows can have different scales.", "Check the byte count on a tiny matrix.|A negative code occupies one signed byte."]
def cases(n):
    row = [0.,-1-n, .5+n/3,3.] if n%4 else [0.,0.,0.,0.]
    return [[row],[[[.125,[1,-4,0]], [.5,[127,1,-7]]],[.2*n,.3,2.],[.1,-.5]],
            [[row,[.01,.03,-.2,0.]] if n!=7 else []]]
MUTANTS = [("peak/127","peak/128"),("round(v/scale)","int(v/scale)"),("b+s*sum","s*sum"),('pack("<f",scale)','pack(">f",scale)')]

