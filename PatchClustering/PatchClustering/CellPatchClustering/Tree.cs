using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CellPatchClustering
{
    public class Tree
    {
        public Node Root { get; set; }

        public List<Node> Nodes { get; private set; }

        public Tree()
        {
            Nodes = new List<Node>();
            Root = AddNode();
        }

        internal Node AddNode()
        {
            var nd = new Node();
            nd.Index = Nodes.Count;
            Nodes.Add(nd);
            return nd;
        }

        public class Node
        {
            public IFeature Feature { get; set; }

            public int Index { get; internal set; }

            internal double bestMetric = double.MaxValue;

            public Node Left { get; set; } // false
            public Node Right { get; set; } // true

            public bool IsLeaf { get { return Left == null; } }
        }
    }
}
