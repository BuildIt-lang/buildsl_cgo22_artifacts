all:
	cp ../outputs/ratio.dat .
	gnuplot plot.sh
	ps2pdf -dEPSCrop plot.ps
	pdfcrop --margins '10 10 10 10' plot.pdf
