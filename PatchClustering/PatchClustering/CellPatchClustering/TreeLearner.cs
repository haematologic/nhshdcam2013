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
            tree.Root.Count = patches.Count;
            foreach (var p in patches) p.NodeIndex = 0;
            for (int d = 0; d < depth; d++)
            {
                AddLayer(tree, patches);
            }
            return tree;
        }

        public void AddLayer(Tree tree, List<Patch> patches) {
            // Get a set of possible features
            var features = GetFeatures(patches[0],100);
            var leafNodes = tree.Nodes.Where(nd => nd.IsLeaf).ToList();

            // For each feature, apply to all patches + count trues and falses, compute metric
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
                foreach (var nd in leafNodes)
                {
                    double metric = Math.Abs(trueCount[nd.Index] - (nd.Count - trueCount[nd.Index]));

                    if (metric < nd.bestMetric)
                    {
                        nd.Feature = f;
                        nd.bestMetric = metric;
                    }

                }
            }
            // Split leaf nodes using best feature
            foreach(var nd in leafNodes)
            {
                //Console.WriteLine("Feature for node " + nd.Index + ": " + nd.Feature);
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
                    nd.Right.Count++;
                }
                else
                {
                    p.NodeIndex= nd.Left.Index;
                    nd.Left.Count++;
                }
            }
        }

        Random rnd = new Random(0);
        private List<IFeature> GetFeatures(Patch p,int N)
        {
            var features = new List<IFeature>();
            for (int i = 0; i < N; i++)
            {
                var f = new AbsoluteIntensityFeature();
                f.Sample(rnd,p);
                features.Add(f);
            }
            return features;
        }
    }
}
