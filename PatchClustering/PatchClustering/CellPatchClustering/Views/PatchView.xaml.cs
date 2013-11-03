using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace CellPatchClustering.Views
{
    /// <summary>
    /// Interaction logic for PatchView.xaml
    /// </summary>
    public partial class PatchView : UserControl
    {
        public PatchView()
        {
            InitializeComponent();
        }

        private void UserControl_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            var p = (Patch)DataContext;
            MyImage.Source = p.Cropped;
            MyBorder.LayoutTransform = new RotateTransform(-p.Angle, p.Width / 2, p.Height / 2); 
            //MyImage.Source = p.Image.Bitmap;
            //Canvas.SetLeft(MyImage, -p.Left);
            //Canvas.SetTop(MyImage, -p.Top);
        }
    }
}
