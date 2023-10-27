import csv
import matplotlib.pyplot as plt

CRSfloattime = open('C:/Users/Erik Bjerned/Documents/Git_repos/ABP/P/CRSfloattime.txt')
csvreaderCRSfloattime = csv.reader(CRSfloattime)

CRSfloattime_rows = []
for row in csvreaderCRSfloattime:
    CRSfloattime_rows.append(row)
    
CRSdoubletime = open('C:/Users/Erik Bjerned/Documents/Git_repos/ABP/P/CRSdoubletime.txt')
csvreaderCRSdoubletime = csv.reader(CRSdoubletime)

CRSdoubletime_rows = []
for row in csvreaderCRSdoubletime:
    CRSdoubletime_rows.append(row)
    
    
CCSfloattime = open('C:/Users/Erik Bjerned/Documents/Git_repos/ABP/P/CCSfloattime.txt')
csvreaderCCSfloattime = csv.reader(CCSfloattime)

CCSfloattime_rows = []
for row in csvreaderCCSfloattime:
    CCSfloattime_rows.append(row)
    
CCSdoubletime = open('C:/Users/Erik Bjerned/Documents/Git_repos/ABP/P/CCSdoubletime.txt')
csvreaderCCSdoubletime = csv.reader(CCSdoubletime)

CCSdoubletime_rows = []
for row in csvreaderCCSdoubletime:
    CCSdoubletime_rows.append(row)
    
"""
CCSfloatperf= open('C:/Users/Erik Bjerned/Documents/Git_repos/ABP/P/CCSfloatperf.txt')
csvreaderCCSfloatperf = csv.reader(CCSfloatperf)

CCSfloatperf_rows = []
for row in csvreaderCCSfloatperf:
    CCSfloatperf_rows.append(row)
print(CCSfloatperf_rows)
    
CRSfloatperf = open('C:/Users/Erik Bjerned/Documents/Git_repos/ABP/P/CRSfloatperf.txt')
csvreaderCRSfloatperf = csv.reader(CRSfloatperf)

CRSfloatperf_rows = []
for row in csvreaderCRSfloatperf:
    CRSfloatperf_rows.append(row)



"""
def column(matrix, i):
    return[float(row[i]) for row in matrix]


plt.title("Memory bandwidth")
plt.plot(column(CRSfloattime_rows,1), column(CRSfloattime_rows, 4))
plt.plot(column(CRSdoubletime_rows,1), column(CRSdoubletime_rows, 4))
plt.plot(column(CCSfloattime_rows,1), column(CCSfloattime_rows, 4))
plt.plot(column(CCSdoubletime_rows,1), column(CCSdoubletime_rows, 4))
plt.legend(["CRS float", "CRS double", "CCS float", "CCS double"])
plt.xlabel("N^3 [-]")
plt.ylabel("Memory bandwidth [GB/s]")
plt.xscale("log")
plt.show()

plt.title("Execution time")
plt.plot(column(CRSfloattime_rows,1), column(CRSfloattime_rows, 5))
plt.plot(column(CRSdoubletime_rows,1), column(CRSdoubletime_rows, 5))
plt.plot(column(CCSfloattime_rows,1), column(CCSfloattime_rows, 5))
plt.plot(column(CCSdoubletime_rows,1), column(CCSdoubletime_rows, 5))
plt.legend(["CRS float", "CRS double", "CCS float", "CCS double"])
plt.xlabel("N^3 [-]")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Time [s]")
plt.show()


plt.clf()

plt.plot(column(CRSfloatperf_rows,1), column(CRSfloatperf_rows, 5))
plt.plot(column(CRSfloatperf_rows,1), column(CRSfloatperf_rows, 6))
plt.plot(column(CRSfloatperf_rows,1), column(CRSfloatperf_rows, 7))
plt.ylabel("Time [s]")
plt.xlabel("N^3 [-]")
plt.xscale("log")
plt.legend(["Matvec", "Dot-prod", "Linear"])
plt.title("Execution time breakdown CRS")
plt.xscale("log")
plt.show()


plt.clf()

plt.plot(column(CCSfloatperf_rows,1), column(CCSfloatperf_rows, 5))
plt.plot(column(CCSfloatperf_rows,1), column(CCSfloatperf_rows, 6))
plt.plot(column(CCSfloatperf_rows,1), column(CCSfloatperf_rows, 7))
plt.ylabel("Time [s]")
plt.xlabel("N^3 [-]")
plt.legend(["Matvec", "Dot-prod", "Linear"])
plt.title("Execution time breakdown CRS")
plt.xscale("log")
plt.show()
