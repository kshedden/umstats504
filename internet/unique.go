/*
unique takes a one-hour flowtuple file and calculates several statistics
on the traffic within one-minute blocks.

The usage is:

> unique [flowtuple file name] [results directory name]
*/

package main

import (
	"compress/gzip"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"strings"

	"github.com/kshedden/flowtuple"
)

var (
	// Total packets per minute
	counts []int

	// UDP packets per minute
	udpCounts []int

	// TCP packets per minute
	tcpCounts []int

	// Distinct IP source addresses per minute
	sources []int

	// Packets per port per minute
	ports [][]int
)

func main() {

	if len(os.Args) != 3 {
		print("Usage: unique [flowtuple file name] [output directory]\n")
		os.Exit(1)
	}

	fname := os.Args[1]
	outdir := os.Args[2]
	if err := os.MkdirAll(outdir, os.ModePerm); err != nil {
		panic(err)
	}

	counts = make([]int, 60)
	udpCounts = make([]int, 60)
	tcpCounts = make([]int, 60)
	sources = make([]int, 60)
	ports = make([][]int, 60)
	for k := 0; k < 60; k++ {
		ports[k] = make([]int, 65536)
	}

	fid, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	gid, err := gzip.NewReader(fid)
	if err != nil {
		panic(err)
	}
	defer gid.Close()

	lf, err := os.Create("testlog.txt")
	if err != nil {
		panic(err)
	}
	logger := log.New(lf, "", log.Ltime)

	ftr := flowtuple.NewFlowtupleReader(gid).SetLogger(logger)

	var frec flowtuple.FlowRec

	// Loop over the intervals (one minute time blocks)
	for {
		err := ftr.ReadIntervalHead()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}

		sourcesSeen := make(map[int]bool)

		for {
			err := ftr.ReadClassHead()
			if err == io.EOF {
				break
			} else if err != nil {
				panic(err)
			}

			for {
				err := ftr.ReadRec(&frec)
				if err == io.EOF {
					break
				} else if err != nil {
					panic(err)
				}

				// Total traffic per minute
				counts[ftr.Inum()]++

				// TCP and UDP traffic per minute
				switch frec.Protocol {
				case 6:
					tcpCounts[ftr.Inum()]++
				case 17:
					udpCounts[ftr.Inum()]++
				}

				// Unique sources per minute
				s := int(frec.SrcIP)
				if !sourcesSeen[s] {
					sources[ftr.Inum()]++
					sourcesSeen[s] = true
				}

				// Ports per minute
				ports[ftr.Inum()][frec.DstPort]++
			}

			err = ftr.ReadClassTail()
			if err == io.EOF {
				break
			} else if err != nil {
				panic(err)
			}
		}

		err = ftr.ReadIntervalTail()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
	}

	// Write the packet counts by minute, for all traffic, TCP traffic, and UDP traffic.  Also
	// save the number of unique source IP addresses seen per minute.
	outname := path.Base(fname)
	outname = strings.Replace(outname, "ucsd-nt.anon.", "", 1)
	outname = strings.Replace(outname, "flowtuple.cors.gz", "csv", 1)
	outname = path.Join(outdir, outname)
	out, err := os.Create(outname)
	if err != nil {
		panic(err)
	}
	defer out.Close()
	io.WriteString(out, "Minute,Packets,Sources,UDP,TCP\n")
	for k := 0; k < 60; k++ {
		out.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n", k, counts[k], sources[k], udpCounts[k], tcpCounts[k]))
	}

	// Write the packet counts per port
	outname = path.Base(fname)
	outname = strings.Replace(outname, "ucsd-nt.anon.", "", 1)
	outname = strings.Replace(outname, "flowtuple.cors.gz", "dports.csv.gz", 1)
	outname = path.Join(outdir, outname)
	out, err = os.Create(outname)
	if err != nil {
		panic(err)
	}
	defer out.Close()
	w := gzip.NewWriter(out)
	defer w.Close()
	for k := 0; k < 60; k++ {
		var x []string
		for _, y := range ports[k] {
			x = append(x, fmt.Sprintf("%d", y))
		}
		w.Write([]byte(strings.Join(x, ",")))
		w.Write([]byte("\n"))
	}
}
