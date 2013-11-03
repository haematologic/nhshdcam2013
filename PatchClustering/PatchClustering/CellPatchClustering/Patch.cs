using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace CellPatchClustering
{
    public class Patch
    {
        public Image Image { get; set; }

        CroppedBitmap cropped;
        internal CroppedBitmap Cropped
        {
            get
            {
                if (cropped == null)
                {
                    cropped = new CroppedBitmap(Image.Bitmap, new Int32Rect(Left+NudgeX, Top+NudgeY, Width, Height));
                }
                return cropped;
            }
        }

        int[][] hist;
        public int[][] Histogram
        {
            get {
                if (hist == null) hist = ComputeHistogram();
                return hist;
            }
        }

        private int[][] ComputeHistogram()
        {
            int[][] h = new int[3][];
            for (int i = 0; i < h.Length; i++) h[i] = new int[8];
            int radius = Width / 2;
            for (int y = 0; y < Height; y++)
            {
                int oy = y - Height / 2;
                for (int x = 0; x < Width; x++)
                {
                    int ox = x - radius;
                    if ((ox * ox + oy * oy) > (radius * radius)) continue; // outside circle
                    var pix = Image.Pixels[Top + NudgeY + y, Left + NudgeX + x];
                    h[0][(pix.Red * h.Length) / 256]++;
                    h[1][(pix.Green * h.Length) / 256]++;
                    h[2][(pix.Blue * h.Length) / 256]++;
                }
            }
            return h;
        }

        public int Top { get; set; }
        public int Left { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }

        public int Angle { get; set; }

        public int NudgeX { get; set; }
        public int NudgeY { get; set; }

        public int NodeIndex { get; set; }
    }
}
