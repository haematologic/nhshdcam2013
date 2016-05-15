using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CellPatchClustering
{
    /// <summary>
    /// A cluster of patches.
    /// </summary>
    public class Cluster
    {
        public int Count { get; set; }

        public List<Patch> SamplePatches { get; set; }
    }
}
