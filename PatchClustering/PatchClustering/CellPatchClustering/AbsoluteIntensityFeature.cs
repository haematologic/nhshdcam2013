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
        public int LowerThreshold { get; set; }  // between 0 and 255
        public int UpperThreshold { get; set; }  // between 0 and 255

        public byte Channel { get; set; } // between 0 (red) and 2 (blue)

        /// <summary>
        /// Computes the feature.
        /// </summary>
        /// <param name="p"></param>
        /// <returns></returns>
        public bool ComputeFeature(Patch p)
        {
            double angleInRads = 2 * Math.PI * p.Angle / 360.0;
            var cos = Math.Cos(angleInRads);
            var sin = Math.Sin(angleInRads);
            double rotx = OffsetX * cos - OffsetY * sin;
            double roty = OffsetX * sin + OffsetY * cos;
            // todo: check that this doesn't go outside of the patch.
            int tx = p.Left + p.NudgeX+ + p.Width / 2 + (int)Math.Round(rotx);
            int ty = p.Top + p.NudgeY+ p.Height / 2 + (int)Math.Round(roty);

            var px = p.Image.Pixels[ty, tx];
            byte val = px.Red;
            if (Channel == 1) val = px.Green; else if (Channel == 2) val=px.Blue;
            return (LowerThreshold <= val) && (val<UpperThreshold);
        }

        public override string ToString()
        {
            return "AbsoluteInt["+LowerThreshold+"<("+Channel+","+OffsetX+","+OffsetY+")<"+UpperThreshold+"]";
        }

        public void Sample(Random rnd, Patch p)
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
            LowerThreshold = rnd.Next(256);
            UpperThreshold = rnd.Next(256);
            if (LowerThreshold > UpperThreshold)
            {
                var dummy = LowerThreshold;
                LowerThreshold = UpperThreshold;
                UpperThreshold = dummy;
            }
            UpperThreshold = 256;
            Channel = (byte)rnd.Next(3);
        }
    }
}
