import os
import matplotlib.pyplot as plt

## Rerun with align option

directory = "C:/Users/Erik Bjerned/Documents/Git_repos/ABP/L1/raw"
count = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    print(f)
    if os.path.isfile(f):

        f = f.replace("\\", "/")

        data = []
        
        
        with open(f, "r") as file:
            for line in file:
                split_line = line.split()
                data.append(split_line)
            
            file.close()
        #print(data) 
        size_vec = [float(line[4]) for line in data]
        mem_vec = [float(line[-2]) for line in data]
        mupds_vec = [float(line[-5]) for line in data]
        #print(mem_vec)
        plt.xscale("log")
        
        plt.plot(size_vec, mem_vec)
        if count == 2:
            mem_512 = mem_vec.copy()
            size_512 = size_vec.copy()
            mupds_float = mupds_vec.copy()
        if count == 4:
            plt.title("Block size")
            plt.legend(["1", "256", "512", "768", "1024"])
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Vector size N [-]")
            #plt.axes((512, 1e8, 0, 256))
            plt.grid()
            plt.show()
            plt.savefig("Blocksize.png")
            
        elif count == 5:
            plt.clf()
            plt.xscale("log")
            plt.title("Memory bandwidth for different data types")
            plt.plot(size_512, mem_512)
            
            plt.plot(size_vec, mem_vec)
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Vector size N [-]")
            plt.legend(["Float", "Double"])
            #plt.axes((512, 1e8, 0, 256))
            plt.grid()
            plt.show()
            plt.savefig("ThroughGB.png")
            plt.clf()
            plt.xscale("log")
            print(mupds_float)
            plt.plot(size_512, mupds_float)
            
            plt.title("Number of updates for different data types")
            plt.plot(size_vec, mupds_vec)
            plt.legend(["Single", "Double"])
            plt.ylabel("MUPD/s[s^-1]")
            plt.xlabel("Vector size N [-]")
            plt.grid()
            plt.show()
            plt.savefig("ThroughMUPDS.png")
            
        elif count == 7:
            mem_O3 = mem_vec.copy()
            size_O3 = size_vec.copy()
            plt.title("Optimization flags")
            plt.xscale("log")
            plt.legend(["O2", "O3"])
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Vector size N [-]")
            plt.grid()
            plt.show()
            plt.savefig("Flags.png")
        elif count == 8:
            plt.clf()
            plt.title("Local AMD R3 3600 against UPPMAX")
            plt.xscale("log")
            plt.plot(size_vec, mem_vec)
            plt.plot(size_O3, mem_O3)
            plt.legend(["Local", "UPPMAX"])
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Vector size N [-]")
            #plt.axes((512, 1e8, 0, 256))
            plt.grid()
            plt.show()
            plt.savefig("CPULocalUPP.png")
            plt.clf()
        elif count == 10:
            
            plt.xscale("log")
            plt.title("Memory alignment -O2, UPPMAX")
            plt.legend(["Non-aligned", "Aligned"])
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Vector size N [-]")
            #plt.axes((512, 1e8, 0, 256))
            plt.grid()
            plt.show()
            plt.savefig("AlignUPP.png")
            plt.clf()
        elif count == 12:
            plt.xscale("log")
            plt.title("Memory alignment -O2, AMD R3 3600")
            plt.legend(["Non-aligned", "Aligned"])
            plt.ylabel("Memory throughput [GB/s]")
            plt.xlabel("Vector size N [-]")
            #plt.axes((512, 1e8, 0, 256))
            plt.grid()
            plt.savefig("AlignLoc.png")
            plt.show()
            
            
        
            
            
            
    count += 1