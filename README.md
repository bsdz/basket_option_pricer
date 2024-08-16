# basket_option_pricer - Basket Option Pricer - C++ Library with Python Binding

Price and calculate greeks for basket options.

Derivation and implementation notes available [here](https://github.com/bsdz/docs/blob/8f78d8524d8a4ade4ba871627bb06698ba0cbc94/papers/Basket%20%26%20Asian%20Option%20Pricing.pdf).

Features

* Python binding

## License

This library is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public
License and is intended for personal use only. It cannot be used in a Bank, Hedgefund, Commodity House etc 
without prior permission from author (Blair Azzopardi).

## Set up

Use conan to build and prepare the dependencies (use clang profile for improved debugging).

```bash
conan install . --output-folder=build/Debug --build=missing --settings=build_type=Debug
conan install . --output-folder=build/Release --build=missing --settings=build_type=Release
```

To specify a different conan profile use switch, e.g. "--profile=clang".


Make the make files, use Release,

```bash
pushd build/Debug
cmake ../.. -DCMAKE_BUILD_TYPE=Debug
popd
pushd build/Release
cmake ../.. -DCMAKE_BUILD_TYPE=Release
popd
```

Once make files are made, one can build the code with (do this after code changes):

```bash
cmake --build . 
```

To clean build

```bash
cmake --build . --target clean
```


Run the test suite:

```bash
cd ./src_test
ctest
```

or directly to view logging:

```bash
./bop_yabte | less
```

## Usage

See accompanying scripts and tests.
