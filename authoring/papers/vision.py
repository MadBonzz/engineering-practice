"""Original small-scale, CPU-runnable vision architecture exercises."""
from textwrap import dedent


def source(code):
    return dedent(code).strip() + "\n"


def part(title, file, funcs, requirement, *hints):
    return dict(title=title, file=file+".py", funcs=funcs.split(), requirement=requirement, hints=list(hints))


def case(n, fn, args, title):
    return dict(part=n, fn=fn, args=args, title=title)


def example(request, expected, why):
    return dict(input=request, output=expected, why=why)


def paper(slug,title,ref,url,mission,inputs,outputs,constraints,scope,modules,parts,examples,checks,duration,level="Intermediate"):
    return dict(slug="paper-"+slug,title=title,group="Vision architectures",level=level,duration=duration,
                mission=mission,inputs=inputs,outputs=outputs,constraints=constraints,scope=scope,
                source=[dict(title=ref,url=url)],modules={k+".py":source(v) for k,v in modules.items()},
                parts=parts,examples=examples,checks=checks)


def traffic(part_no, requests):
    return [case(part_no,"gateway.handle",[r],name) for name,r in requests]


PAPERS=[]

res_examples=[
    example(dict(image=[[2,1,3]],stride=1,gain=-2,shortcut=None),
            dict(features=[[0,1,0]],mean=1/3),
            "The horizontal difference detector produces [2, 0, 2] after its ReLU. The second convolution scales it by −2. Adding the original pixels before applying the final ReLU gives [0, 1, 0]."),
    example(dict(image=[[-2,9],[7,5]],stride=2,gain=0,shortcut=-3),
            dict(features=[[6]],mean=6),
            "The sampled input is −2. A zero learned branch leaves the projected shortcut, which contributes 6 before the output ReLU.")
]
PAPERS.append(paper("resnet","ResNet · Route identity and projected shortcuts","He et al. · Deep Residual Learning for Image Recognition","https://arxiv.org/abs/1512.03385",
    "A small vision runtime exports a residual stage for two spatial resolutions. Implement both paths of the stage so a stride change samples the same input coordinates on each path. A negative residual can suppress an otherwise strong detector.",
    "image is a nonempty rectangular H×W feature plane. stride is 1 or 2. gain scales the second convolution. shortcut is null for identity or a scalar weight for a learned 1×1 projection.",
    "Return features, the residual stage's output plane, and mean, its global spatial average.",
    "Sample anchor pixels at row and column multiples of stride. Treat an out-of-bounds left neighbor as zero. Output shape is ceil(H/stride) × ceil(W/stride). Identity is requested only at stride 1.",
    "This is a single post-activation ResNet-v1 basic block on one input channel, followed by global average readout. The first 3×3 filter uses the center and left taps, the second uses its center tap, and batch normalization is folded to identity. Full ResNet stacks many wider blocks before a learned classifier.",
    dict(residual_path='''
        def residual(image, stride, gain):
            """Return gain × ReLU(center − left) at each strided anchor, shape ceil(H/s)×ceil(W/s)."""
            return [[gain * max(0, row[c] - (row[c-1] if c else 0))
                     for c in range(0, len(row), stride)]
                    for row in image[::stride]]
    ''', shortcuts='''
        def shortcut_plane(image, stride, scale):
            """Return the H×W identity when scale is None, else a strided 1×1 scalar projection."""
            if scale is None:
                return [row[:] for row in image]
            return [[scale * row[c] for c in range(0,len(row),stride)] for row in image[::stride]]
    ''', stage='''
        from residual_path import residual
        from shortcuts import shortcut_plane

        def block(image, stride, gain, scale):
            """Compute ReLU(shortcut + learned branch) pointwise on the two aligned output planes."""
            path = residual(image, stride, gain)
            skip = shortcut_plane(image, stride, scale)
            return [[max(0, a+b) for a,b in zip(x,y)] for x,y in zip(path,skip)]
    ''',gateway='''
        from stage import block

        def handle(request):
            out = block(request["image"],request["stride"],request["gain"],request["shortcut"])
            total = sum(map(sum,out))
            return {"features":out,"mean":total/(len(out)*len(out[0]))}
    '''),
    [part("Run the learned branch","residual_path","residual","Emulate two convolutions with a ReLU between them. The first computes center minus left at each anchor. The second multiplies by gain.","The left neighbor is one original pixel away, including at stride 2.","Keep negative values produced by the second layer."),
     part("Align the shortcut","shortcuts","shortcut_plane","Project the original input with the same stride as the residual branch when a scalar scale is supplied. Preserve signed activations.","A numeric zero is a valid projection weight.","The shortcut never observes the hidden activations of the learned branch."),
     part("Fuse the branches","stage","block","Add the learned and shortcut outputs at matching coordinates, then clamp the sum at zero.","Try negative gain on a descending ramp.","Return a fresh output plane.")],
    res_examples,
    [case(1,"residual_path.residual",[[[1,5,2,8]],2,2],"Left tap is adjacent to the strided anchor"),case(1,"residual_path.residual",[[[-3,-1],[4,0]],1,-1],"Preserve the sign of gain"),
     case(2,"shortcuts.shortcut_plane",[[[-2,4],[3,-5]],1,None],"Signed identity"),case(2,"shortcuts.shortcut_plane",[[[1,2,3],[4,5,6],[7,8,9]],2,0],"Zero projection with odd shape"),
     case(3,"stage.block",[[[3,2,4]],1,-1,None],"Residual suppresses a positive shortcut"),case(3,"stage.block",[[[-2,-1]],1,1,-2],"ReLU after addition")]+traffic(3,[("Stride-two projected odd image",dict(image=[[2,0,1],[4,4,4],[-1,2,7]],stride=2,gain=-.5,shortcut=2)),("Flat input",dict(image=[[1,1],[1,1]],stride=1,gain=3,shortcut=None)),("Negative projection",dict(image=[[2,-1,4]],stride=1,gain=.25,shortcut=-1))]),"40–55 min"))

def dense_input(image,growth,weights):return dict(image=image,growth=growth,compression=weights)
PAPERS.append(paper("densenet","DenseNet · Grow and compress a feature bank","Huang et al. · Densely Connected Convolutional Networks","https://arxiv.org/abs/1608.06993",
    "A mobile encoder repeatedly creates one new feature channel and retains every earlier channel for reuse. Wire a dense block into a transition layer. The transition must see features in exactly the same channel order as the exported weights.",
    "image is the initial H×W plane. growth contains one vector per layer, with 1+i weights at zero-based layer i. compression has one weight for each channel in the final bank.",
    "Return widths, the initial and successive channel counts, and pooled, the compressed one-channel feature map after 2×2 average pooling.",
    "H and W are at least 2. Each layer forms a pointwise weighted sum over ReLU of every existing channel and appends the signed result. The transition applies the same preactivation rule before mixing. Pooling uses stride 2 and ignores incomplete border windows.",
    "Implement the channel-growth invariant of a DenseNet block and its compression/downsampling transition. Each growth filter uses only its center spatial tap with growth rate one. Batch normalization is identity, with a single starting plane and a single transition output.",
    dict(feature_bank='''
        def append_feature(bank, weights):
            """Return a new C+1 channel bank, appending sum_c weights[c] * ReLU(bank[c]) at every pixel."""
            h, w = len(bank[0]), len(bank[0][0])
            plane = [[sum(k*max(0,channel[r][c]) for k,channel in zip(weights,bank))
                      for c in range(w)] for r in range(h)]
            return bank + [plane]
    ''',transition='''
        def compress_pool(bank, weights):
            """Mix C rectified channels into one plane and average complete 2×2 tiles with stride 2."""
            h,w = len(bank[0]), len(bank[0][0])
            def mixed(r,c):
                return sum(k*max(0,channel[r][c]) for k,channel in zip(weights,bank))
            return [[sum(mixed(r+i,c+j) for i in range(2) for j in range(2))/4
                     for c in range(0,w-1,2)] for r in range(0,h-1,2)]
    ''', gateway='''
        from feature_bank import append_feature
        from transition import compress_pool

        def handle(request):
            bank = [request["image"]]
            widths = [len(bank)]
            for kernel in request["growth"]:
                bank = append_feature(bank,kernel)
                widths.append(len(bank))
            return {"widths":widths,"pooled":compress_pool(bank,request["compression"])}
    '''),
    [part("Append without discarding history","feature_bank","append_feature","Build the next feature from all previous channels in oldest-first order. Preserve those channels and append the new plane.","Rectify each input channel before weighting.","A negative sum remains in the bank until a later layer reads it."),
     part("Compress at the transition","transition","compress_pool","Use an independent 1×1 preactivated projection across the whole bank. Average each complete, non-overlapping 2×2 tile.","Compression can weight old and new channels differently.","Reducing width and height is separate from reducing channels.")],
    [example(dense_input([[1,-2],[3,0]],[[2],[0,1]],[1,1,-.5]),dict(widths=[1,2,3],pooled=[[2]]),"On positive input x the two new features each equal 2x. The transition yields x + 2x − x = 2x. Its four pixels are 2, 0, 6, 0, which average to 2."),
     example(dense_input([[-1,2,100],[1,-1,100]],[],[2]),dict(widths=[1],pooled=[[1.5]]),"An empty block still has the original input. The transition sees 0, 2, 1, 0 in its only full tile and doubles their mean. The last column is discarded.")],
    [case(1,"feature_bank.append_feature",[[[[1,-2]],[[-1,3]]],[2,-1]],"Order and preactivation"),case(1,"feature_bank.append_feature",[[[[2,1],[3,4]]],[-2]],"Preserve signed growth"),
     case(2,"transition.compress_pool",[[[[1,-2],[3,4]],[[0,1],[2,0]]],[-1,2]],"Signed transition"),case(2,"transition.compress_pool",[[[[1,1,8],[1,1,8],[9,9,9]]],[1]],"Incomplete tiles")]+traffic(2,[("Cross-layer feature reuse",dense_input([[1,2],[3,4]],[[2],[-1,2],[2,0,-1]],[1,2,3,4])),("Negative growth",dense_input([[1,-3],[2,-5]],[[-2]],[0,4])),("Rectangular compression",dense_input([[1,2,3,4],[4,3,2,1]],[[.5]],[2,-1]))]),"40–55 min"))
ID3=[[0,0,0],[0,1,0],[0,0,0]]
LEFT=[[0,0,0],[1,0,0],[0,0,0]]
EDGE=[[0,0,0],[-1,1,0],[0,0,0]]
BLUR=[[0,.25,0],[.25,0,.25],[0,.25,0]]

def vgg_input(image, stages):return dict(image=image,stages=stages)

PAPERS.append(paper("vgg","VGG · Execute a configurable 3×3 stage stack","Simonyan and Zisserman · Very Deep Convolutional Networks for Large-Scale Image Recognition","https://arxiv.org/abs/1409.1556",
    "An image classifier stores its feature extractor as a stack of stages. A mistaken placement of pooling silently changes the receptive field. Implement its feature extractor and track each stage's shape for the classifier head.",
    "image is a single feature plane H×W. stages is a list of nonempty lists of 3×3 filter matrices. Each filter is followed by ReLU, and each entire stage ends with one max pool.",
    "Return shapes as a list of [height,width] pairs after each stage's pool and features as the final feature plane.",
    "Use 3×3 cross-correlation at unit stride, with one zero of padding on each edge and no bias. The 2×2 max pool has stride 2 and drops incomplete windows. Each spatial dimension is at least 2 before every pool.",
    "This mini VGG reproduces sequential same-resolution 3×3 filters, intermediate ReLUs and stage-boundary pooling. It uses one channel across stages and returns the feature extractor output rather than the original classification head.",
    dict(spatial_filter='''
        def filter_relu(image, kernel):
            """Compute unit-stride zero-padded 3×3 cross-correlation plus ReLU, preserving H×W."""
            h,w=len(image),len(image[0])
            result=[]
            for row in range(h):
                line=[]
                for col in range(w):
                    total=0
                    for kr in range(3):
                        for kc in range(3):
                            r,c=row+kr-1,col+kc-1
                            if 0<=r<h and 0<=c<w:
                                total+=image[r][c]*kernel[kr][kc]
                    line.append(max(0,total))
                result.append(line)
            return result
    ''',stages='''
        from spatial_filter import filter_relu

        def pool2(image):
            """Max over complete 2×2 windows at stride 2, returning floor(H/2)×floor(W/2)."""
            return [[max(image[r+i][c+j] for i in range(2) for j in range(2))
                     for c in range(0,len(image[0])-1,2)] for r in range(0,len(image)-1,2)]

        def run_stack(image, specs):
            """Return (features, shapes) after all filters and one pool per stage, in listed order."""
            shapes=[]
            for filters in specs:
                for kernel in filters:
                    image=filter_relu(image,kernel)
                image=pool2(image)
                shapes.append([len(image),len(image[0])])
            return [image,shapes]
    ''',gateway='''
        from stages import run_stack

        def handle(request):
            feature,shape=run_stack(request["image"],request["stages"])
            return {"shapes":shape,"features":feature}
    '''),
    [part("Implement the padded filter","spatial_filter","filter_relu","Return one activation at each input coordinate. Kernel entry [0][0] multiplies the upper-left neighbor.","The zero-padding contributes zero when a neighbor is outside the image.","Clamp each filter's result before evaluating the next filter."),
     part("Execute the stage specification","stages","pool2 run_stack","Apply each stage's filters in sequence, pool once, and record the post-pool dimensions. run_stack returns [final_plane, shapes].","A stage containing two filters has only one pool.","Include only completed windows when an edge has odd length.")],
    [example(vgg_input([[1,4,2,7],[0,3,6,5]],[[EDGE,ID3]]),
             dict(shapes=[[1,2]],features=[[3,5]]),"The first filter returns [1,3,0,5] and [0,3,3,0]. The identity filter preserves these activations. Max pooling the left and right tiles produces [3,5]."),
     example(vgg_input([[-1,-2,100],[-3,-4,100]],[[ID3]]),dict(shapes=[[1,1]],features=[[0]]),"ReLU clears the four negative values in the complete tile. The large values lie in the incomplete right column.")],
    [case(1,"spatial_filter.filter_relu",[[[1,2,3]],LEFT],"Asymmetric filter orientation"),case(1,"spatial_filter.filter_relu",[[[2],[-1]],BLUR],"Border padding and ReLU"),
     case(2,"stages.pool2",[[[-1,-4,3,7],[-6,-2,4,2],[-9,0,0,100]]],"Rectangular max-pool geometry"),case(2,"stages.run_stack",[[[1,4],[2,3]],[[LEFT,EDGE]]],"Both filters run before pooling")]+traffic(2,[("Two resolution changes",vgg_input([[4,1,8,7],[3,-2,4,1],[0,5,4,8],[-1,-1,2,3]],[[EDGE,ID3],[LEFT]])),("Spatial averaging before pooling",vgg_input([[1,0],[0,1]],[[BLUR]])),("Filter order on a wide image",vgg_input([[3,1,4,5,9],[2,6,5,3,6]],[[LEFT,EDGE]]))]),"45–60 min"))

def unet_input(image,up,blend,bias=0):return dict(image=image,upsample_kernel=up,blend=blend,bias=bias)

PAPERS.append(paper("unet","U-Net · Align a decoder with a cropped skip","Ronneberger et al. · U-Net: Convolutional Networks for Biomedical Image Segmentation","https://arxiv.org/abs/1505.04597",
    "A tile segmenter combines a coarse context feature with a fine encoder feature. Implement the expanding path and stitch the encoder skip to the correct coordinates. A one-pixel offset changes which objects are marked foreground.",
    "image is an H×W scalar encoder feature grid. The supplied contracting stage retains this skip, removes one border pixel on each side and max-pools the center. upsample_kernel is a 2×2 learned kernel. blend gives [up_weight, skip_weight] for the pointwise head, and bias is scalar.",
    "Return logits and mask at the decoder resolution. mask contains 1 exactly where the corresponding fused logit is strictly positive.",
    "H and W are at least 4. Pool complete 2×2 windows. Stride-2 transpose convolution stamps its 2×2 kernel once per low-resolution sample with no overlap or padding. Center-crop the skip. If the excess size is odd, discard the extra pixel on the bottom or right.",
    "This one-level U-Net uses a provided cropped/pooling encoder, a learned up-convolution, a center-cropped skip, and a one-channel segmentation head. It isolates skip alignment and fusion from the original multi-level learned convolution stacks.",
    dict(encoder='''
        def contract(skip):
            """Produce the pooled bottleneck from the valid interior of a supplied encoder feature."""
            core=[row[1:-1] for row in skip[1:-1]]
            return [[max(core[r+i][c+j] for i in range(2) for j in range(2))
                     for c in range(0,len(core[0])-1,2)] for r in range(0,len(core)-1,2)]
    ''',upconv='''
        def expand(low, kernel):
            """Map H×W to 2H×2W by stamping the 2×2 kernel times each low-resolution pixel."""
            h,w=len(low),len(low[0])
            return [[low[r//2][c//2]*kernel[r%2][c%2] for c in range(2*w)] for r in range(2*h)]
    ''',skip_join='''
        def center_crop(skip, height, width):
            """Extract height×width from the center. Floor the top and left offsets."""
            top=(len(skip)-height)//2
            left=(len(skip[0])-width)//2
            return [row[left:left+width] for row in skip[top:top+height]]

        def fuse(up, skip, weights, bias):
            """Crop skip to match up; apply weights[0]*up + weights[1]*cropped_skip + bias per pixel."""
            crop=center_crop(skip,len(up),len(up[0]))
            return [[weights[0]*x+weights[1]*y+bias for x,y in zip(a,b)] for a,b in zip(up,crop)]
    ''',gateway='''
        from encoder import contract
        from upconv import expand
        from skip_join import fuse

        def handle(request):
            skip=request["image"]
            up=expand(contract(skip),request["upsample_kernel"])
            logits=fuse(up,skip,request["blend"],request["bias"])
            return {"logits":logits,"mask":[[int(v>0) for v in row] for row in logits]}
    '''),
    [part("Restore the spatial lattice","upconv","expand","Implement a scalar 2×2 stride-2 transposed convolution. Every bottleneck sample writes its own four output locations.","Associate the first kernel entry with the top-left position of its tile.","Different kernel taps can have opposite signs."),
     part("Align and join the skip","skip_join","center_crop fuse","Center-crop the skip and use the two ordered feature channels as inputs to the segmentation head. Return signed pre-threshold logits.","Compute height and width offsets independently.","The head treats the upsampled feature as channel 0.")],
    [example(unet_input([[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]],[[1,0],[-1,1]],[1,-1]),
             dict(logits=[[3,-2],[-7,0]],mask=[[1,0],[0,0]]),"Pooling [1,2;3,4] yields 4. The up-convolution stamps [[4,0],[-4,4]]. Subtracting the aligned skip gives the four logits. A zero is background."),
     example(unet_input([[1]*5 for _ in range(5)],[[0,0],[0,0]],[0,2],-1),
             dict(logits=[[1,1],[1,1]],mask=[[1,1],[1,1]]),"With zero decoder weights the retained skip supplies the entire prediction. Its unit values become logits of 1 after projection and bias.")],
    [case(1,"upconv.expand",[[[2,-1]],[[1,2],[3,4]]],"Adjacent tiles and kernel orientation"),case(1,"upconv.expand",[[[1],[3]],[[0,-1],[2,0]]],"Vertical tile ordering"),
     case(2,"skip_join.center_crop",[[[r*10+c for c in range(6)] for r in range(5)],2,3],"Unequal odd crop offsets"),case(2,"skip_join.fuse",[[[1,2]],[[0,3,4],[5,6,7]],[.5,-1],2],"Blending unequal shapes")]+traffic(2,[("Odd rectangular alignment",unet_input([[r*7+c for c in range(7)] for r in range(5)],[[0,1],[2,0]],[1,-1],2)),("Negative context",unet_input([[-i-1]*4 for i in range(4)],[[1,2],[1,0]],[2,1])),("Decoder-only projection",unet_input([[1,2,3,4,5,6]]*6,[[1,-1],[0,0]],[1,0],-1))]),"50–65 min"))

def alex_input(image,filters,radius=1,k=1,alpha=1,beta=1):
    return dict(image=image,filters=filters,radius=radius,k=k,alpha=alpha,beta=beta)

PAPERS.append(paper("alexnet","AlexNet · Normalize competing feature channels","Krizhevsky, Sutskever and Hinton · ImageNet Classification with Deep Convolutional Neural Networks","https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
    "An inference team is porting an AlexNet feature stage whose strongest detector sometimes changes after response normalization. Compute normalization across neighboring channels and feed each normalized plane into an overlapping max pool.",
    "image is H×W. Each [weight,bias] in filters defines a supplied pointwise feature detector followed by ReLU. radius is the number of neighboring channels on each side. k, alpha and beta are normalization constants.",
    "Return pooled, a channels-first C×out_h×out_w tensor of normalized and max-pooled activations.",
    "H and W are at least 3. For channel c at a fixed pixel, output a_c / (k + alpha × sum_j a_j²)^beta over j from c−radius through c+radius, clipped to existing channels. There is no window-size divisor. Use valid 3×3 max pooling, stride 2. k is positive and alpha and beta are nonnegative.",
    "This reproduces local response normalization and overlapping pooling from the 2012 model. A supplied bank of small pointwise detectors replaces the large first convolution so every activation can be inspected. The exercise ends at the pooled feature maps.",
    dict(detectors='''
        def activations(image, filters):
            """Produce C rectified feature planes using the exported scale and bias per detector."""
            return [[[max(0,weight*pixel+bias) for pixel in row] for row in image]
                    for weight,bias in filters]
    ''',response_norm='''
        def normalize(planes, radius, k, alpha, beta):
            """Apply local response normalization to C×H×W at fixed row and column, inclusive channel window."""
            result=[]
            for index,plane in enumerate(planes):
                start=max(0,index-radius)
                end=min(len(planes),index+radius+1)
                output=[]
                for r,row in enumerate(plane):
                    line=[]
                    for c,value in enumerate(row):
                        energy=sum(planes[j][r][c]**2 for j in range(start,end))
                        line.append(value/(k+alpha*energy)**beta)
                    output.append(line)
                result.append(output)
            return result
    ''',overlap_pool='''
        def pool3(plane):
            """Return maxima over complete 3×3 windows with top-left corners spaced two pixels apart."""
            return [[max(plane[r+i][c+j] for i in range(3) for j in range(3))
                     for c in range(0,len(plane[0])-2,2)] for r in range(0,len(plane)-2,2)]
    ''',gateway='''
        from detectors import activations
        from response_norm import normalize
        from overlap_pool import pool3

        def handle(request):
            z=activations(request["image"],request["filters"])
            normalized=normalize(z,request["radius"],request["k"],request["alpha"],request["beta"])
            return {"pooled":[pool3(plane) for plane in normalized]}
    '''),
    [part("Implement cross-channel competition","response_norm","normalize","At each location compute the denominator from the original rectified channels, including the center channel. Preserve tensor shape.","Use neighboring channels at one pixel, not neighboring pixels.","At channel boundaries clip the window and use the same alpha."),
     part("Pool overlapping neighborhoods","overlap_pool","pool3","Pool each normalized channel independently. A 3×5 plane has two overlapping windows across its width.","Start the second window at column 2.","Drop any window extending past the right or bottom edge.")],
    [example(alex_input([[0,1,0],[0,0,0],[0,0,0]],[[1,0],[-1,1]]),
             dict(pooled=[[[.5]],[[.5]]]),"Exactly one of the two detectors fires with value 1 at each pixel. With radius 1, every denominator is 1 + 1² = 2. Both channels have a positive response somewhere in the pool."),
     example(alex_input([[0,0,3,0,0],[0,0,0,0,0],[0,0,0,0,0]],[[1,0]],0,1,0,1),
             dict(pooled=[[[3,3]]]),"Alpha zero disables competition. The central value 3 belongs to both windows because their starting columns differ by two.")],
    [case(1,"response_norm.normalize",[[[[1]],[[3]],[[2]]],1,1,1,1],"Use clipped channel neighborhoods"),case(1,"response_norm.normalize",[[[[2,1],[0,4]]],0,2,.25,.5],"Keep pixel energies separate"),
     case(2,"overlap_pool.pool3",[[[0,0,2,0,1],[3,0,0,0,0],[0,0,0,0,4]]],"Overlapping horizontal windows"),case(2,"overlap_pool.pool3",[[[2,0,3],[1,0,0],[0,10,0],[0,0,0]]],"Drop incomplete final window")]+traffic(2,[("Three detector neighborhood",alex_input([[0,1,2],[3,2,1],[0,0,1]],[[1,0],[0,2],[-1,1]],1,2,.1,.75)),("Wide neighborhood",alex_input([[1,0,0,1,4]]*3,[[1,1],[0,1]],4,1,.5,1)),("Each channel alone",alex_input([[-1,0,1]]*5,[[1,1],[-1,1]],0,2,1,.5))]),"35–50 min"))

from math import sqrt

def vit_input(image,patch,projection,cls,positions):
    return dict(image=image,patch=patch,projection=projection,cls=cls,positions=positions)

half=0.5/sqrt(1+1e-6)
normalized_half=half/sqrt(half*half+1e-6)
PAPERS.append(paper("vit","Vision Transformer · Assemble a patch-token classifier","Dosovitskiy et al. · An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale","https://arxiv.org/abs/2010.11929",
    "Ship a tiny image classifier whose first token reads evidence from non-overlapping patches. Implement the patch sequence, attention and residual encoder, then send the final class token to a two-label head. A transposed patch or a misplaced class position changes every later computation.",
    "image is a grayscale H×W plane divisible by patch size p. projection is 2×p², cls is the learned length-2 class token, and positions supplies one length-2 learned embedding for every token, including cls.",
    "Return patch_count, the final pre-head cls_state vector after both residual additions, and logits from the supplied head applied to the last normalized class vector.",
    "Enumerate patches and pixels inside each patch in row-major order. The one-head encoder uses full attention including self-attention. Q, K and V use identity projections. Divide scores by sqrt(2). Use supplied row normalization, fixed ReLU MLP and class head. Inputs may be rectangular.",
    "This is an end-to-end miniature ViT inference path. It uses width 2, a single pre-normalized block and head, grayscale patches, fixed example weights, and a ReLU MLP in place of the original wider GELU MLP. It retains learned-position addition, the class token and both residual paths.",
    dict(patch_tokens='''
        def tokenize(image, size, projection, cls, positions):
            """Return [CLS, patch_1,...] after adding positions. Each patch row-major vector maps to dimension 2."""
            tokens=[list(cls)]
            for r in range(0,len(image),size):
                for c in range(0,len(image[0]),size):
                    patch=[image[r+i][c+j] for i in range(size) for j in range(size)]
                    tokens.append([sum(w*x for w,x in zip(row,patch)) for row in projection])
            return [[v+p for v,p in zip(token,pos)] for token,pos in zip(tokens,positions)]
    ''',attention_core='''
        import math

        def attend(tokens):
            """Map N×2 tokens to N×2 softmax-weighted sums using identity Q/K/V, dot products divided by sqrt(2)."""
            result=[]
            for query in tokens:
                scores=[sum(a*b for a,b in zip(query,key))/math.sqrt(2) for key in tokens]
                maximum=max(scores)
                weights=[math.exp(s-maximum) for s in scores]
                norm=sum(weights)
                result.append([sum(w*t[d] for w,t in zip(weights,tokens))/norm for d in range(2)])
            return result
    ''',fixed_weights='''
        from math import sqrt

        def norm(row):
            """Unit gain, zero offset layer norm over two coordinates using population variance and eps=1e-6."""
            mean=sum(row)/2
            scale=sqrt(sum((x-mean)**2 for x in row)/2 + 1e-6)
            return [(x-mean)/scale for x in row]

        def mlp(row):
            """Fixed one-unit hidden ReLU followed by the output vector [1, -1]."""
            activation=max(0,row[0])
            return [activation,-activation]

        def head(row):
            """Two class logits. The supplied projection is the 2×2 identity."""
            return list(row)
    ''',image_encoder='''
        from attention_core import attend
        from fixed_weights import norm, mlp

        def encode(tokens):
            """Return N×2 hidden states with z=x+attention(LN(x)), then y=z+MLP(LN(z)) rowwise."""
            updates=attend([norm(row) for row in tokens])
            middle=[[v+u for v,u in zip(row,update)] for row,update in zip(tokens,updates)]
            return [[v+u for v,u in zip(row,mlp(norm(row)))] for row in middle]
    ''',gateway='''
        from patch_tokens import tokenize
        from image_encoder import encode
        from fixed_weights import head,norm

        def handle(request):
            tokens=tokenize(request["image"],request["patch"],request["projection"],request["cls"],request["positions"])
            state=encode(tokens)[0]
            return {"patch_count":len(tokens)-1,"cls_state":state,"logits":head(norm(state))}
    '''),
    [part("Start from image patches","patch_tokens","tokenize","Make one projected embedding per non-overlapping patch, prepend cls and add position vectors elementwise. Use projection rows as output features.","A rectangular grid can have different numbers of patch rows and columns.","Position 0 belongs to the class token."),
     part("Run full token attention","attention_core","attend","Compute a row-normalized attention distribution per query, including the class token. Implement a numerically stable softmax.","The denominator uses all keys for the current query.","Subtract the maximum score before exponentiation."),
     part("Assemble the encoder and readout","image_encoder","encode","Call the supplied norm and mlp on individual rows. Normalize before the attention branch, then add its update to the original tokens. Repeat pre-normalization and residual addition for the MLP branch.","The residual path carries the unnormalized hidden state.","The gateway reads only the final state at token index zero.")],
    [example(vit_input([[-1]],1,[[1],[-1]],[0,0],[[0,0],[0,0]]),
             dict(patch_count=1,cls_state=[-half,half],logits=[-normalized_half,normalized_half]),
             "The class query is zero, so it averages its zero value and the normalized patch [−a,a], with a=1/√1.000001. Its MLP update vanishes because the first normalized coordinate is negative. Final layer normalization rescales [−a/2,a/2] to the two logits."),
     example(vit_input([[1,2],[3,4]],2,[[1,0,0,0],[1,0,0,0]],[3,3],[[0,0],[1,1]]),
             dict(patch_count=1,cls_state=[3,3],logits=[0,0]),
             "Both the class and patch vectors are constant across their two coordinates. Normalization maps both to zero. Both residual updates are zero, leaving the class state at [3,3] and its final normalized logits at zero.")],
    [case(1,"patch_tokens.tokenize",[[[1,2,3,4],[5,6,7,8]],2,[[0,1,0,0],[0,0,1,0]],[1,2],[[1,1],[0,1],[1,0]]],"Raster order on rectangular patch grid"),case(1,"patch_tokens.tokenize",[[[1],[2]],1,[[-1],[2]],[0,0],[[3,3],[1,-1],[2,-2]]],"Position zero belongs to CLS"),
     case(2,"attention_core.attend",[[[0,0],[2,-1],[-2,3]]],"Zero query averages all values"),case(2,"attention_core.attend",[[[1000,1000],[1000,-1000]]],"Stable large dot products"),
     case(3,"image_encoder.encode",[[[1,-1],[2,2]]],"Two successive residual paths"),case(3,"image_encoder.encode",[[[0,4],[-1,2],[4,4]]],"Layer normalization per token")]+traffic(3,[("Learned class offset",vit_input([[1,0]],1,[[2],[-1]],[2,-1],[[1,2],[0,0],[1,0]])),("Four nonidentical patch embeddings",vit_input([[0,1],[2,4]],1,[[-1],[1]],[0,0],[[1,-1],[0,0],[1,0],[0,2],[-2,0]])),("Two larger patches",vit_input([[1,-1,3,0],[2,1,0,4]],2,[[1,-1,0,1],[0,1,2,0]],[-1,1],[[1,1],[1,0],[0,-1]]))]),"65–90 min","Advanced"))

# These probes distinguish common almost-correct implementations from the paper mechanics.
adversarial = {
    "paper-resnet": [
        case(3,"stage.block",[[[5,1,8,2,3]],2,-1,1],
             "A strided filter reads the original left neighbor, not the previous strided anchor"),
        case(3,"stage.block",[[[-1,-5,1,4]],1,-2,-1],
             "A projected shortcut can be negative and must be added before the output ReLU"),
    ],
    "paper-densenet": [
        case(1,"feature_bank.append_feature",[[[[-2]],[[3]],[[-4]]],[1,-1,1]],
             "Rectify old channels individually and keep a negative newly computed channel"),
        case(2,"gateway.handle",[dense_input([[2,0,7],[0,0,7]],[[5],[3,-1]],[1,0,-2])],
             "A transition may select the oldest feature after growth and discard the ragged border"),
    ],
    "paper-vgg": [
        case(1,"spatial_filter.filter_relu",[[[1,2],[3,4]],[[1,0,0],[0,0,0],[0,0,0]]],
             "Upper-left tap distinguishes unflipped cross-correlation from convolution"),
        case(2,"stages.run_stack",[[[0,5,0,0],[0,0,0,0]],[[LEFT,LEFT]]],
             "Two spatial shifts move an activation across a pool boundary before any downsampling"),
    ],
    "paper-unet": [
        case(2,"skip_join.center_crop",[[[r*10+c for c in range(8)] for r in range(4)],3,3],
             "Independent offsets when only one axis has more than one surplus pixel"),
        case(2,"gateway.handle",[unet_input([[9,9,9,9],[9,4,1,9],[9,2,3,9],[9,9,9,9]],[[1,0],[0,1]],[1,-1])],
             "An exactly zero segmentation logit stays background after the ordered skip join"),
    ],
    "paper-alexnet": [
        case(1,"response_norm.normalize",[[[[2]],[[3]],[[4]],[[7]]],1,1,1,1],
             "One-sided window at channel zero excludes the far channel and uses unnormalized original energy"),
        case(2,"overlap_pool.pool3",[[[0]*5,[0]*5,[0,0,9,0,0],[0]*5,[0]*5]],
             "The shared center contributes to all four overlapping windows"),
    ],
    "paper-vit": [
        case(1,"patch_tokens.tokenize",[[[1,2],[3,4],[8,7],[6,5]],2,[[0,1,0,0],[0,0,1,0]],[1,-1],[[2,3],[-2,1],[4,-3]]],
             "Vertical patches retain row-major internal order and distinct learned positions"),
        case(3,"gateway.handle",[vit_input([[5,-1]],1,[[0],[0]],[0,0],[[0,0],[1,-1],[-1,1]])],
             "Opposite position-only values exactly cancel at a zero class query"),
    ],
}
for lab in PAPERS:
    lab["checks"].extend(adversarial[lab["slug"]])
