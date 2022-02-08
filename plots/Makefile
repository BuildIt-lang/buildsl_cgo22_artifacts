all:
	cp ../outputs/ratio.dat .
	gnuplot plot.sh
	ps2pdf -dPDFSETTINGS=/prepress -dEmbedAllFonts=true   plot.ps
	pdf270 plot.pdf
	mv plot-rotated270.pdf plot.pdf
	pdfcrop --margins '10 10 10 10' plot.pdf
