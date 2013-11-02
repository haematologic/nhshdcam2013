using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CellPatchClustering
{
    public class TreeLearner
    {

        public Tree Learn(int depth, List<Patch> patches)
        {
            var tree = new Tree();
            for (int d = 0; d < depth; d++)
            {
                AddLayer(tree, patches);
            }
            return tree;
        }

        public void AddLayer(Tree tree, List<Patch> patches) {
            // Get a set of possible features
            var features = GetFeatures(patches[0],100);

            // For each feature, apply to all patches + count trues and falses, compute metric
            int[] count = new int[tree.Nodes.Count];
            foreach(var p in patches) count[p.NodeIndex]++;
            foreach (var f in features)
            {
                int[] trueCount = new int[tree.Nodes.Count];
                foreach (var p in patches)
                {
                    if (f.ComputeFeature(p))
                    {
                        trueCount[p.NodeIndex]++;
                    }
                }
                for (int i = 0; i < count.Length; i++)
                {
                    var nd = tree.Nodes[i];
                    if (!nd.IsLeaf) continue;

                    double metric = Math.Abs(trueCount[i] - (count[i] - trueCount[i]));

                    if (metric < nd.bestMetric)
                    {
                        nd.Feature = f;
                        nd.bestMetric = metric;
                    }
                }
            }
            // Split leaf nodes using best feature
            for (int i = 0; i < count.Length; i++)
            {
                var nd = tree.Nodes[i];
                if (!nd.IsLeaf) continue;
                nd.Left = tree.AddNode();
                nd.Right = tree.AddNode();
            }
            // Update which node each patch is at
            foreach (var p in patches)
            {
                var nd = tree.Nodes[p.NodeIndex];
                if (nd.Feature.ComputeFeature(p))
                {
                    p.NodeIndex = nd.Right.Index;
                }
                else
                {
                    p.NodeIndex= nd.Left.Index;
                }
            }
        }

        Random rnd = new Random(0);
        private List<IFeature> GetFeatures(Patch p,int N)
        {
            var features = new List<IFeature>();
            for (int i = 0; i < N; i++)
            {
                var f = new AbsoluteIntensityFeature
                {
                     OffsetX = rnd.Next(p.Width),
                     OffsetY = rnd.Next(p.Height),
                     Threshold = rnd.Next(256)
                };
                features.Add(f);
            }
            return features;
        }
    }
}
