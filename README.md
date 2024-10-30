### R+ is a DNA storage transcoding strategy developed by BGI-research. Briefly, it can provide a direct mapping refence between expanded molecular alphabet and N-nary digits in the absence of high-performance transcoding algorithm at present.
### R+ strategy is based on the following table:
<img src="rotational_table.png">


# Environment Configuration
***
### The kit is developed by Python 3.10.
### Additionally, following packages:
### <p style="line-height: 0;"> &diams; sys </p>
### <p style="line-height: 0;"> &diams; struct </p>
### <p style="line-height: 0;"> &diams; re </p>
### <p style="line-height: 0;"> &diams; subprocess </p>
### <p style="line-height: 0;"> &diams; itertools </p>
### <p style="line-height: 0;">
### and extension packages:
### <p style="line-height: 0;"> &diams; numpy </p>
### <p style="line-height: 0;"> &diams; tqdm </p>
### <p style="line-height: 0;"> &diams; loguru </p>
### <p style="line-height: 0;"> &diams; biopython </p>
### <p style="line-height: 0;">
### are required.

# Kit Tree Diagram
***
    ├── examples                          // Test module
    │    ├── files                        // Test files
    │    │    ├── Mona Lisa.jpg           // Mona Lisa.jpg
    │    ├── output                       // Generated files from pipeline
    │    ├── test_mona_lisa.py            // Run R+ using Mona Lisa.jpg
    ├── R+
    │    ├── utils                        // Util module
    │    │    ├── log.py                  // Output the logs in console
    │    │    ├── tools.py                // Set of functions used in experimental validation and simulation evaluation
    │    ├── pipeline.py                  // Main calling function
    │    ├── scheme.py                    // R+
    ├── README.md                         // Description document of kit

# Introduction of R+
***
### R+ is an algorithm that describes rotational transcoding rules in a generated form.
### The users could install this package by "pip install R-plus". When you have finished installing the package, the sample program in [folder](./examples) could be run to make sure the package is correct.
### We suggest using Python IDE (e.g., PyCharm) to run the program.
### Example of command lines are as follows:
```python
    python
    >>> from R_plus import scheme, pipeline
    >>> # encoding process
    >>> pipeline.encode(
    >>>     algorithm=scheme.RotaitonPlus(virtual_letter="A", N_values=4, extra_letters"M", need_logs=True, need_monitor=True),
    >>>     input_file="../files/Mona_Lisa.bmp",
    >>>     output_file="../output/Mona_Lisa.txt",
    >>>     ins_xor=True,
    >>> )
    >>> # decoding process
    >>> pipeline.decode(
    >>>     algorithm=scheme.RotaitonPlus(virtual_letter="A", N_values=4, extra_letters"M", need_logs=True, need_monitor=True),
    >>>     input_file="./output/Mona_Lisa.txt",
    >>>     output_file="./output/output_Mona_Lisa.bmp",
    >>>     del_xor=True,
    >>> )

