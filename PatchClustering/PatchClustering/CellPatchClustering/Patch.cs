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
                    cropped = new CroppedBitmap(Image.Bitmap, new Int32Rect(Left, Top, Width, Height));
                }
                return cropped;
            }
        }

        public int Top { get; set; }
        public int Left { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }

        public int NodeIndex { get; set; }
    }
}
