import numpy as np

def main():
	directory = "/projects/onebusaway/BakerNiedMLProject/data"
	serviceName = "intercitytransit"
	routeName = "route13"
	filespec = "{}/{}_{}_*.txt".format(directory, serviceName, routeName);
	for files in os.listdir(filespec):
		print files

if __name__ == '__main__':
	main()
