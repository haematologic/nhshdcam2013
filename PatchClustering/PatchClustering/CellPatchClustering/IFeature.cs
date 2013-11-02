using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CellPatchClustering
{
    public interface IFeature
    {

        bool ComputeFeature(Patch p);
    }
}
