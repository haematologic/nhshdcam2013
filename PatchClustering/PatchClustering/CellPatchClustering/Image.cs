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

        internal Pixel[,] Pixels;

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

            Pixels = new Pixel[h,w];
            Bitmap.Lock();
            unsafe
            {
                var pixels = (byte*)Bitmap.BackBuffer;
                int i=0;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++, i += 4)
                    {
                        Pixels[y, x] = new Pixel(pixels[i+2],pixels[i+1],pixels[i]);
                    }
                }
            }
           // Bitmap.AddDirtyRect(new Int32Rect(0, 0, w, h));
            Bitmap.Unlock();
        }

        public void AddPatches(List<Patch> patches,int offsetx, int offsety,int patchWidth,int patchHeight)
        {
            int w = Bitmap.PixelWidth;
            int h = Bitmap.PixelHeight;
            for (int y = 0; y <= h - patchHeight; y += offsety)
            {
                for (int x = 0; x <= w - patchWidth; x += offsetx)
                {
                    var p = new Patch { Left = x, Top = y, Width = patchWidth, Height = patchHeight, Image = this };
                    patches.Add(p);
                }
            }
        }

        public struct Pixel
        {
            public byte Red;
            public byte Green;
            public byte Blue;

            public Pixel(byte r, byte g, byte b)
            {
                Red = r;
                Green = g;
                Blue = b;
            }
        }
    }
}
