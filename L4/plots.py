import os
import matplotlib.pyplot as plt
## Rerun with align option

f = "times.txt"
count = 0
data = [];
test_data = {}

def column(matrix, i):
    return[float(row[i]) for row in matrix]


with open(f, "r") as file:
    for line in file:
        if "interval" in line: continue;
        split_line = line.split()
        if (len(split_line) == 1):
            curr_key = split_line[0]
            test_data[curr_key] =[]; 
        else:
            test_data[curr_key].append(split_line);
            
    file.close()

N = column(test_data["GPUTIME"], 6)
"""
plt.xscale("log")
plt.plot(N, column(test_data["FCR"],10))
plt.plot(N, column(test_data["DCR"],10))
plt.plot(N, column(test_data["FGL"],10))
plt.plot(N, column(test_data["DGL"],10))
plt.grid()
plt.ylabel("Million elements per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-right", "Double-cpu-right", "Float-gpu-left", "Double-gpu-left"])
plt.title("Comparison of float and double precision on CPU and GPU")
plt.savefig("FDCGmele.png")

plt.clf()
plt.xscale("log")
plt.plot(N, column(test_data["FCR"],12))
plt.plot(N, column(test_data["DCR"],12))
plt.plot(N, column(test_data["FGL"],12))
plt.plot(N, column(test_data["DGL"],12))
plt.grid()
plt.ylabel("GFLOPs per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-right", "Double-cpu-right", "Float-gpu-left", "Double-gpu-left"])
plt.title("Comparison on CPU and GPU")
plt.savefig("FDCGgflops.png")

plt.clf()
plt.xscale("log")
plt.plot(N, column(test_data["FCR"],14))
plt.plot(N, column(test_data["DCR"],14))
plt.plot(N, column(test_data["FGL"],14))
plt.plot(N, column(test_data["DGL"],14))
plt.grid()
plt.ylabel("GB per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-right", "Double-cpu-right", "Float-gpu-left", "Double-gpu-left"])
plt.title("Comparison on CPU and GPU")
plt.savefig("FDCGgb.png")



plt.clf()
plt.xscale("log")
plt.plot(N, column(test_data["FCL"],10))
plt.plot(N, column(test_data["DCL"],10))
plt.plot(N, column(test_data["FGL"],10))
plt.plot(N, column(test_data["DGL"],10))
plt.grid()
plt.ylabel("Million elements per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-left", "Double-cpu-left", "Float-gpu-left", "Double-gpu-left"])
plt.title("Comparison on CPU and GPU, Kokkos::LayoutLeft")
plt.savefig("FDCGLmele.png")

plt.clf()
plt.xscale("log")
plt.plot(N, column(test_data["FCL"],12))
plt.plot(N, column(test_data["DCL"],12))
plt.plot(N, column(test_data["FGL"],12))
plt.plot(N, column(test_data["DGL"],12))
plt.grid()
plt.ylabel("GFLOPs per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-left", "Double-cpu-left", "Float-gpu-left", "Double-gpu-left"])
plt.title("Comparison on CPU and GPU, Kokkos::LayoutLeft")
plt.savefig("FDCGLgflops.png")

plt.clf()
plt.xscale("log")
plt.plot(N, column(test_data["FCL"],14))
plt.plot(N, column(test_data["DCL"],14))
plt.plot(N, column(test_data["FGL"],14))
plt.plot(N, column(test_data["DGL"],14))
plt.grid()
plt.ylabel("GB per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-left", "Double-cpu-left", "Float-gpu-left", "Double-gpu-left"])
plt.title("Comparison on CPU and GPU, Kokkos::LayoutLeft")
plt.savefig("FDCGLgb.png")


plt.clf()
plt.xscale("log")
plt.plot(N, column(test_data["FCR"],10))
plt.plot(N, column(test_data["DCR"],10))
plt.plot(N, column(test_data["FGR"],10))
plt.plot(N, column(test_data["DGR"],10))
plt.grid()
plt.ylabel("Million elements per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-right", "Double-cpu-right", "Float-gpu-right", "Double-gpu-right"])
plt.title("Comparison on CPU and GPU, Kokkos::LayoutRight")
plt.savefig("FDCGRmele.png")

plt.clf()
plt.xscale("log")
plt.plot(N, column(test_data["FCR"],12))
plt.plot(N, column(test_data["DCR"],12))
plt.plot(N, column(test_data["FGR"],12))
plt.plot(N, column(test_data["DGR"],12))
plt.grid()
plt.ylabel("GFLOPs per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-right", "Double-cpu-right", "Float-gpu-right", "Double-gpu-right"])
plt.title("Comparison on CPU and GPU, Kokkos::LayoutRight")
plt.savefig("FDCGRgflops.png")

plt.clf()
plt.xscale("log")
plt.plot(N, column(test_data["FCR"],14))
plt.plot(N, column(test_data["DCR"],14))
plt.plot(N, column(test_data["FGR"],14))
plt.plot(N, column(test_data["DGR"],14))
plt.grid()
plt.ylabel("GB per second [s^-1]")
plt.xlabel("No. elements [-]")
plt.legend(["Float-cpu-right", "Double-cpu-right", "Float-gpu-right", "Double-gpu-right"])
plt.title("Comparison on CPU and GPU, Kokkos::LayoutRight")
plt.savefig("FDCGRgb.png")

plt.clf()
plt.xscale("log")
plt.yscale("log")
plt.plot(N, column(test_data["FCL"],17))
plt.plot(N, column(test_data["FCL"],8))
plt.plot(N, column(test_data["FGL"],17))
plt.plot(N, column(test_data["FGL"],8))
plt.grid()
plt.ylabel("Time [s]")
plt.xlabel("No. elements [-]")
plt.legend(["Jtime-float-cpu-left", "Ctime-float-cpu-left", "Jtime-float-gpu-left", "Ctime-float-gpu-left"])
plt.title("Time comparison on CPU and GPU, Kokkos::LayoutLeft")
plt.savefig("TL.png")

plt.clf()
plt.xscale("log")
plt.yscale("log")
plt.plot(N, column(test_data["FCL"],17))
plt.plot(N, column(test_data["FCL"],8))
plt.plot(N, column(test_data["FGL"],17))
plt.plot(N, column(test_data["FGL"],8))
plt.grid()
plt.ylabel("Time [s]")
plt.xlabel("No. elements [-]")
plt.legend(["Jtime-float-cpu-right", "Ctime-float-cpu-right", "Jtime-float-gpu-right", "Ctime-float-gpu-right"])
plt.title("Time comparison on CPU and GPU, Kokkos::LayoutRight")
plt.savefig("TR.png")
"""

plt.clf()
plt.xscale("log")
plt.yscale("log")

plt.plot(N, column(test_data["CPUTIME"], 23))
plt.plot(N, column(test_data["CPUTIME"],17))
plt.plot(N, column(test_data["CPUTIME"],8))
plt.plot(N, column(test_data["GPUTIME"],8))
plt.plot(N, column(test_data["CPUTIME"], 20))
plt.grid()
plt.ylabel("Time [s]")
plt.xlabel("No. elements [-]")
plt.legend(["Jinit-cpu", "Jtransfer-cpu", "Ccompute-cpu", "Ccompute-gpu", "Ctransfer-cpu"])
plt.title("Time comparison on CPU and GPU")
plt.savefig("T.png")

