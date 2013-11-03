using System;
using System.Collections.Generic;
using System.IO;
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
            var patches = new List<Patch>();

            if (true)
            {
                //C:\Users\johnw_000\Images\bigger

                string imageFolder = @"C:\Users\johnw_000\Images\imageset1";
                var imfiles = Directory.GetFiles(imageFolder, "*.jpg");
                var images = imfiles.Select(filename => new Image(System.IO.Path.Combine(imageFolder, filename))).ToList();

                // Image im = images[0];
                foreach (var im in images) im.AddPatches(patches, 31, 31, 31, 31);
            }
            else
            {
                string imageFolder = @"C:\Users\johnw_000\Images\bigger";
                var imfiles = Directory.GetFiles(imageFolder, "*.jpg");
                var images = imfiles.Select(filename => new Image(System.IO.Path.Combine(imageFolder, filename))).ToList();

                // Image im = images[0];
                foreach (var im in images.Take(800)) im.AddPatches(patches, 5,5, 47, 47);

            }

            var learner = new TreeLearner();

            Tree tree = null;
            for (int z = 0; z < 4; z++)
            {
                // Learn initial tree
                tree = learner.Learn(10, patches);

                // Learning rotations
                Console.WriteLine("Learning rotations & translations");
                for (int iters = 0; iters < 2; iters++)
                {
                    LearnRotations(patches, tree);
                    Console.WriteLine("Non-empty leaves: " + tree.Nodes.Count(nd => nd.IsLeaf && nd.Count > 0));
                    //LearnTranslations(patches, tree);
                    //Console.WriteLine("Non-empty leaves: " + tree.Nodes.Count(nd => nd.IsLeaf && nd.Count > 0));
                }
                //Console.WriteLine("Learning translations");
                
                //Console.WriteLine("Non-empty leaves: " + tree.Nodes.Count(nd => nd.IsLeaf && nd.Count > 0));
            }

            var clusters = new List<Cluster>();
            foreach (var nd in tree.Nodes)
            {
                if (nd.IsLeaf)
                {
                    var cl = new Cluster
                    {
                        Count = nd.Count,
                        SamplePatches = new List<Patch>()
                    };

                    var leafPatches = patches.Where(p => p.NodeIndex == nd.Index).ToList();
                    for(int i=0;i<leafPatches.Count;i+=5) {
                        cl.SamplePatches.Add(leafPatches[i]);
                        if (cl.SamplePatches.Count > 20) break;
                    }
                    clusters.Add(cl);
                }
            }
            Console.WriteLine("Empty clusters: " + clusters.Count(c => c.Count == 0) + " out of " + clusters.Count);
            MyClusters.ItemsSource = clusters.OrderByDescending(cl => cl.Count);
        }

        private static void LearnRotations(List<Patch> patches, Tree tree)
        {
            for (int iters = 0; iters < 2; iters++)
            {
                // Loop over the patches
                foreach (var p in patches)
                {
                    int bestCount = 0;
                    int bestAngle = 0;
                    var oldnd = p.NodeIndex;
                    var newnd = -1;
                    //   Loop over rotations
                    for (int angle = 0; angle < 360; angle += 5)
                    {
                        p.Angle = angle;
                        //     Apply the tree to the patch
                        var leaf = tree.Apply(p);

                        //     See if bigger, if so store best rotation
                        if (leaf.Count > bestCount)
                        {
                            bestCount = leaf.Count;
                            bestAngle = angle;
                            newnd = leaf.Index;
                        }
                    }
                    //   Move to node for best rotation
                    p.NodeIndex = newnd;
                    p.Angle = bestAngle;
                    tree.Nodes[oldnd].Count--;
                    tree.Nodes[newnd].Count++;
                }
            }
        }

        private static void LearnTranslations(List<Patch> patches, Tree tree)
        {
            int max = 16; int step = 2;
            // Loop over the patches
            foreach (var p in patches)
            {
                int bestCount = 0;
                int bestX = 0; int bestY = 0;
                var oldnd = p.NodeIndex;
                var newnd = -1;
                //   Loop over rotations
                for (int y = -max; y <= max; y += step)
                {
                    for (int x = -max; x <= max; x += step)
                    {
                        p.NudgeX = x;
                        p.NudgeY = y;
                        if ((x + p.Left < 0) || (x + p.Left + p.Width >= p.Image.Bitmap.PixelWidth)) continue;
                        if ((y + p.Top < 0) || (y + p.Top + p.Height >= p.Image.Bitmap.PixelHeight)) continue;

                        //  Apply the tree to the patch
                        var leaf = tree.Apply(p);

                        //  See if bigger, if so store best rotation
                        if (leaf.Count > bestCount)
                        {
                            bestCount = leaf.Count;
                            bestX = x; bestY = y;
                            newnd = leaf.Index;
                        }
                    }
                }
                //   Move to node for best rotation
                p.NodeIndex = newnd;
                p.NudgeX = bestX;
                p.NudgeY = bestY;
                tree.Nodes[oldnd].Count--;
                tree.Nodes[newnd].Count++;
            }
        }


    }
}
