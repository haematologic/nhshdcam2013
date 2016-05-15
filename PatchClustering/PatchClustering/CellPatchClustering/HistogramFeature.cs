using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CellPatchClustering
{
    public class HistogramFeature : IFeature
    {
        public int Bin {get;set;} // 0..number of bins

        public int Threshold { get; set; }

        public byte Channel { get; set; } // between 0 (red) and 2 (blue)

        public bool ComputeFeature(Patch p)
        {
            return p.Histogram[Channel][Bin] > Threshold;
        }

        public override string ToString()
        {
            return "Histogram["+Channel+","+Bin+">"+Threshold+"]";
        }

        public void Sample(Random rnd, Patch p)
        {
            Bin = rnd.Next(p.Histogram[0].Length);
            Threshold = rnd.Next(p.Width * p.Height);
            Channel = (byte)rnd.Next(3);
        }
    }
}
