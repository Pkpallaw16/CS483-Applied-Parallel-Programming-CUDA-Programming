# 🚀 CS483 / ECE408 - Applied Parallel Programming (CUDA Programming)

This repository contains lab assignments and project work completed as part of the **CS483 / ECE408: Applied Parallel Programming** course offered at the University of Illinois. The course focuses on designing and implementing parallel programs using **CUDA** and modern **many-core processors**.

---

## 🧪 Lab Projects

| Lab  | Topic                                      |
|------|--------------------------------------------|
| Lab 0 | Setup & Environment Test                  |
| Lab 1 | Parallel Vector Addition                  |
| Lab 2 | Parallel Matrix Multiplication            |
| Lab 3 | Tiled Matrix Multiplication               |
| Lab 4 | Parallel Reduction                        |
| Lab 5 | Parallel Scan                             |
| Lab 6 | Tiled Parallel Convolution                |
| Lab 7 | Sparse Matrix-Vector Multiplication       |
| Lab 8 | Final Project Setup & Submission Scripts  |


---

## 🚀 Project Milestones

### 🧩 Milestone 1: Naive Convolution
- Implemented 2D convolution using a simple CUDA kernel.
- Used only global memory and measured baseline runtime.

### 🔁 Milestone 2: Matrix Unrolling + GEMM
- Transformed convolution into matrix multiplication via input unrolling.
- Verified correctness and profiled kernel performance.

### ⚙️ Milestone 3: Optimized GPU Convolution
- Applied advanced CUDA optimizations:
  - `Streams`
  - `Shared Memory` tiling / `Tensor Cores`
  - `Kernel Fusion`
- Achieved < 80ms batch inference time for 10,000 images.
- Experimented with additional techniques: `cuBLAS`, `__restrict__`, `loop unrolling`.

---

## 📊 Technologies Used
- **CUDA 12+**
- **C++ / CMake**
- **Nsight Compute / Systems**
- **Slurm + Batch Scripts**
- **UIUC Delta Cluster**

---

## ⚠️ Academic Integrity Notice

> 🚫 **FOR ACADEMIC USE ONLY – DO NOT COPY**

This repository is made public **only for educational purposes**. All code is original work and must not be copied or submitted as part of coursework. Doing so violates academic integrity policies.

### 🚫 Violations include:
- Submitting this work as your own.
- Using this code without proper attribution.
- Collaborating when not authorized.

> The author is not responsible for any academic misconduct resulting from misuse of this repository.

## 👨‍💻 Author

**Pallaw Kumar**  
Graduate Student – Computer Science  
University of Illinois Urbana-Champaign  
[GitHub: @Pkpallaw16](https://github.com/Pkpallaw16)

---

> Feel free to explore, fork, and learn — but use responsibly and ethically.
