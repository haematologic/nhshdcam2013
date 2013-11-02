using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace CellPatchClustering
{
    public class Image
    {
        internal WriteableBitmap Bitmap;

        internal double[,] Blue;

        public Image(string filename)
        {
            BitmapImage bi = new BitmapImage(new Uri(filename));
            Bitmap = new WriteableBitmap(bi);
            ConvertToPixels();
        }

        private void ConvertToPixels()
        {
            int w = Bitmap.PixelWidth;
            int h = Bitmap.PixelHeight;
            int stride = Bitmap.BackBufferStride;
            int size = stride * h;

            Blue = new double[h,w];
            Bitmap.Lock();
            unsafe
            {
                var pixels = (byte*)Bitmap.BackBuffer;
                int i=2;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++, i += 4)
                    {
                        Blue[y, x] = pixels[i];
                    }
                }
            }
           // Bitmap.AddDirtyRect(new Int32Rect(0, 0, w, h));
            Bitmap.Unlock();
        }

        public List<Patch> GetPatches(int offsetx, int offsety,int patchWidth,int patchHeight)
        {
            int w = Bitmap.PixelWidth;
            int h = Bitmap.PixelHeight;
            var patches = new List<Patch>();
            for (int y = 0; y <= h - patchHeight; y += offsety)
            {
                for (int x = 0; x <= w - patchWidth; x += offsetx)
                {
                    var p = new Patch { Left = x, Top = y, Width = patchWidth, Height = patchHeight, Image = this };
                    patches.Add(p);
                }
            }
            return patches;
        }
    }
}
