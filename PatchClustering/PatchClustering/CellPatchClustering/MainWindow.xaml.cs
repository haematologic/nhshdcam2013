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

namespace CellPatchClustering
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Image im = new Image(@"C:\Users\johnw_000\Images\0-49200-8192.jpg");
            DataContext = im.Bitmap;

            //IFeature f = new AbsoluteIntensityFeature{ OffsetX=0, OffsetY=0, Threshold=128 };
            var patches = im.GetPatches(5, 5, 25, 25);
            
            var learner = new TreeLearner();
            var tree = learner.Learn(7,patches);
            var clusters = new List<List<Patch>>();
            foreach(var nd in tree.Nodes)
            {
                if (nd.IsLeaf)
                {
                    var cl = patches.Where(p => p.NodeIndex == nd.Index).Take(20).ToList();
                    clusters.Add(cl);
                }
            }
                MyClusters.ItemsSource = clusters;
      }
    }
}
