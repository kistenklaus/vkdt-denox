# vkdt-denox
denox codegen for vkdt

### Compilation with denox and codegeneration.
1. First we define a pytorch model. For example:
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding="same", dtype=torch.float16)

    def forward(self, input):
        return self.conv(input)
```
2. Write weights into the pytorch model.
There exist a bunch of different way to achieve this if you trained the model in pytorch, 
the model should already contian the correct weights.

3. Export model from pytorch to onnx.
```python
example_input = torch.ones(1, 3, 1920, 1080, dtype=torch.float16)
net =  Net()
program = torch.onnx.export(
    net,
    (example_input,),
    dynamic_shapes={
        "input": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}
    },
    input_names=["input"],
    output_names=["output"])
program.save("net.onnx")
```
4. Populate denox database. 
From the command line run:
```bash
denox net.onnx --db gpu.db --populate \
    --optimize-for="H:1080,W:1920" \
    --input-shape=H:W:C  \
    --input-type=f16 \
    --input-layout=HWC \
    --input-storage=SSBO \
    --output-type=f16 \
    --output-layout=HWC \
    --output-storage=SSBO \
    --device="*NVIDIA*" \
    --target-env=vulkan1.4 \
    --spirv-no-debug-info \
    --no-spirv-non-semantic-debug-info \
    --spirv-optimize \
    --ffusion \
    --fcoopamt \
    --fmemory-concat
```
If the database file "gpu.db" does not exist it will create and compile all 
possible dispatch configurations into this file. If the gpu.db file already
exists this will add new entries into the database.

5. Benchmarking the database.
```bash
denox --bench=gpu.db --timeout=15min --min-samples=10
```
This will benchmark the gpu.db file for as long as you specify, the results will be stored
in the gpu.db file, if the database already contains measurements results that match your 
criteria (i.e. min-samples > 10) then this exists immediately.

6. Compile model
```bash
denox net.onnx --db gpu.db -o net.dnx\
    --optimize-for="H:1080,W:1920" \
    --input-shape=H:W:C  \
    --input-type=f16 \
    --input-layout=HWC \
    --input-storage=SSBO \
    --output-type=f16 \
    --output-layout=HWC \
    --output-storage=SSBO \
    --device="*NVIDIA*" \
    --target-env=vulkan1.4 \
    --spirv-no-debug-info \
    --no-spirv-non-semantic-debug-info \
    --spirv-optimize \
    --ffusion \
    --fcoopamt \
    --fmemory-concat
```
This will compile the model into the net.dnx file.
This file then contains all weights and all shader binaries 
as well as a list of dispatches which have to be executed in order.

7. Run the codegeneration
```bash
vkdt-denox net.dnx -o pipe/modules/ --module-name=denox-conv \
    --preprocessing=input.comp --post-processing=output.comp
```
At this point vkdt-denox does not know where it will take the values for the dynamic 
variables i.e. "H" and "W".





