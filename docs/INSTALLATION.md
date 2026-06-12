## Installation

### 1. Preparation
Before starting, open a terminal and navigate to your desired working directory:
```bash
cd ~/path/to/directory
```

### 2. Clone the Repositories
- Clone the GEDPaths repository:
  ```bash
  git clone git@github.com:mlai-bonn/GEDPaths.git
  ```
- Clone the libGraph library in the same directory (otherwise you have to change the paths in the code):
  ```bash
  git clone git@github.com:mlai-bonn/libGraph.git
  ```

### 3. Install GEDLIB (Unix)
See [gedlib](https://github.com/dbblumenthal/gedlib) for details.
- Navigate to the external folder of libGraph:
  ```bash
  cd libGraph/external
  ```
- Clone the GEDLIB repository:
  ```bash
  git clone git@github.com:dbblumenthal/gedlib.git
  ```
- Install CMake:
  ```bash
  sudo apt-get install cmake
  ```
- Install Doxygen:
  ```bash
  sudo apt-get install doxygen
  ```
- Only under MacOS: Install OpenMP

#### Dependencies for GEDLIB
- **Eigen 3.4.0** (navigate to `libGraph/external/gedlib/ext` before running these commands):
  ```bash
  git clone https://gitlab.com/libeigen/eigen.git
  cd eigen
  git checkout 3.4.0
  cd ..
  ```
- **Boost 1.89.0** (navigate to `libGraph/external/gedlib/ext` before running these commands):
  ```bash
  wget https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.bz2
  tar --bzip2 -xf boost_1_89_0.tar.bz2
  rm boost_1_89_0.tar.bz2
  mv boost_1_89_0 boost
  ```

- For the exact solvers please install GUROBI ([Gurobi 12.0.3 Installation instructions](#install-gurobi-1203-linux) below)
- Install GEDLIB using:
  ```bash
  python install.py [--gurobi <GUROBI_ROOT>]
  ```

---

### Install Gurobi 12.0.3 (Linux)

1. Register for a Gurobi account and obtain an academic license (if eligible):
   https://www.gurobi.com/downloads/
2. Download Gurobi 12.0.3 for Linux from the official website:
   https://www.gurobi.com/downloads/gurobi-optimizer-eula/
3. Extract the downloaded archive:
   ```bash
   tar -xvf gurobi12.0.3_linux64.tar.gz
   ```
4. Move the extracted folder to your desired location (e.g., ~/opt):
   ```bash
   mv gurobi12.0.3_linux64 ~/opt/
   ```
5. Set up your Gurobi license:
   - Run the license setup tool and follow the prompts:
     ```bash
     ~/opt/gurobi12.0.3_linux64/bin/grbgetkey <YOUR-LICENSE-KEY>
     ```
   - This will create a license file in your home directory (~/gurobi.lic).
6. Add Gurobi to your environment variables (add to ~/.bashrc or ~/.zshrc):
   ```bash
   export GUROBI_HOME=~/opt/gurobi12.0.3_linux64
   export PATH="${GUROBI_HOME}/bin:${PATH}"
   export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
   ```
   - Reload your shell configuration:
     ```bash
     source ~/.bashrc
     ```
7. (Optional) Test your installation:
   ```bash
   grbprobe
   ```
   - This should print your license and system information.

For more details, see the official [Gurobi Installation Guide](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer).
