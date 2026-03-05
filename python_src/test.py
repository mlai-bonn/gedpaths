import os


def main():
    # load GEDDataset from torch_geometric.datasets
    # import molhiv dataset from ogb package
    from ogb.graphproppred import PygGraphPropPredDataset
    dataset = PygGraphPropPredDataset(name="ogbg-code2")
    print(f"Dataset length: {len(dataset)}")

    # get all graphs up to size 16 nodes and get their y value and write it to targets.txt under Results/Mappings/Precomputed/molhiv-16
    output_dir = os.path.join("Results", "Mappings", "Precomputed", "code2-22")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "targets.txt"), "w") as f:
        for i in range(len(dataset)):
            data = dataset[i]
            if data.num_nodes <= 22:
                # write y list entries as space-separated values
                y_str = " ".join(str(y) for y in data.y)
                f.write(f"{y_str}\n")

if __name__ == "__main__":
    main()