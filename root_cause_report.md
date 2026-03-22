## 🛑 Root Cause Analysis Report
### 1. Numerical Evidence
- **Total Stabilization Attempts:**  / 10
- **Final dt reached:** 
### 2. Convergence Audit
### 3. Recommended Fix (Pedigree Update)
- 🟢 **Fix B:** Your iterations (00001000012017) are sufficient. Decrease initial `time_step` in input JSON; the simulation is physically jumping too far.
