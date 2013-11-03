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
        public int Threshold { get; set; }  // between 0 and 255

        public byte Channel { get; set; } // between 0 (red) and 2 (blue)

        /// <summary>
        /// Computes the feature.
        /// </summary>
        /// <param name="p"></param>
        /// <returns></returns>
        public bool ComputeFeature(Patch p)
        {
            double angleInRads = 2 * Math.PI * p.Angle / 360.0;
            double rotx = OffsetX * Math.Cos(angleInRads) - OffsetY * Math.Sin(angleInRads);
            double roty = OffsetX * Math.Sin(angleInRads) + OffsetY * Math.Cos(angleInRads);
            // todo: check that this doesn't go outside of the patch.
            int tx = p.Left + p.Width / 2 + (int)Math.Round(rotx);
            int ty = p.Top + p.Height / 2 + (int)Math.Round(roty);

            var px = p.Image.Pixels[ty, tx];
            if (Channel == 0) return px.Red > Threshold;
            if (Channel == 1) return px.Green > Threshold;
            return px.Blue > Threshold;
        }

        public override string ToString()
        {
            return "AbsoluteInt[("+Channel+","+OffsetX+","+OffsetY+")>"+Threshold+"]";
        }

        internal void Sample(Random rnd, Patch p)
        {
            bool inCircle = false;
            int halfWidth = p.Width / 2;
            do
            {
                // todo: work out about even vs odd size patches
                OffsetX = rnd.Next(p.Width) - halfWidth;
                OffsetY = rnd.Next(p.Height) - p.Height/2;
                inCircle = (OffsetX * OffsetX + OffsetY * OffsetY)<=(halfWidth*halfWidth);
            } while (!inCircle);
            Threshold = rnd.Next(256);
            Channel = (byte)rnd.Next(3);
        }
    }
}
