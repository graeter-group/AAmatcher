#%%
from pathlib import Path

# import sys
# module_path = str(Path(__file__).parent.parent.parent.parent)
# if module_path not in sys.path:
#     sys.path.append(module_path)

from AAmatcher.xyz2res.model_ import Representation
from dgl import load_graphs
import torch



if __name__ == "__main__":

    model = Representation(256,out_feats=1,n_residuals=2,n_conv=1)

    graphs, _ = load_graphs("./tmp/pdb_ds.bin")

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    #%%

    def predict(scores):
        return torch.where(scores >=0, 1, 0)

    def accuracy(pred, ref):
        t = torch.where(pred==ref, 1, 0)
        acc = t.sum() / torch.ones_like(t).sum()
        return acc.item()

    #%%
    epochs = 20
    model = model.to("cuda")
    for epoch in range(epochs):
        tl = 0
        preds = []
        refs = []
        for g in graphs:
            optim.zero_grad()
            g = g.to("cuda")
            g = model(g)
            pred = g.ndata["h"][:,0]
            ref = g.ndata["c_alpha"].float()
            loss = loss_fn(pred, ref)
            loss.backward()
            optim.step()
            preds.append(predict(pred))
            refs.append(ref)
            tl += loss
        preds = torch.cat(preds,dim=0)
        refs = torch.cat(refs,dim=0)
        print(f"train loss: {tl.item():.4f}, train acc: {accuracy(preds, refs):.4f}")
    # %%
    model.eval()
    model = model.to("cpu")
    torch.save(model.state_dict(), Path(__file__).parent.parent / Path("model2.pt"))
    # %%
