using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CellPatchClustering
{
    class AbsoluteIntensityFeature : IFeature
    {
        public int OffsetX { get; set; } // between 0 and patchWidth-1
        public int OffsetY { get; set; } // between 0 and patchHeight-1
        public double Threshold { get; set; }  // between 0 and 255

        /// <summary>
        /// Computes the feature.
        /// </summary>
        /// <param name="p"></param>
        /// <returns></returns>
        public bool ComputeFeature(Patch p)
        {
            int tx = p.Left+ OffsetX ;
            int ty = p.Top + OffsetY;
            return p.Image.Blue[ty,tx] > Threshold;
        }

        public override string ToString()
        {
            return "AbsoluteInt[("+OffsetX+","+OffsetY+")>"+Threshold+"]";
        }
    }
}
