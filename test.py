"""
GEARS Model 测试脚本
"""
import numpy as np
import anndata as ad
import pandas as pd
from perturblab.model.gears import GearsConfig, GearsModel
from perturblab.data import PerturbationData

def create_test_data(n_genes=50, n_cells=200):
    """创建简单的测试数据"""
    print("创建测试数据...")
    
    # 使用 GO 数据库中真实存在的基因名称
    # 这些基因已经验证存在于 GO 数据库中（通过 check_go_genes.py 验证）
    # 使用常见的、在单细胞数据中经常出现的基因
    common_genes = [
        'ACTB', 'GAPDH', 'TUBB', 'CD4', 'CD8A', 'IL2', 'TNF', 'IFNG', 'TP53', 'MYC',
        'BRCA1', 'BRCA2', 'EGFR', 'VEGFA', 'TGFB1', 'IL6', 'IL10', 'IL1B', 'NFKB1', 'STAT3',
        'JUN', 'FOS', 'MAPK1', 'AKT1', 'PIK3CA', 'PTEN', 'RB1', 'MDM2', 'BCL2', 'CASP3',
        'CDKN1A', 'CDKN2A', 'CCND1', 'CCNE1', 'CDK4', 'CDK6', 'E2F1', 'MYCN', 'KRAS', 'NRAS',
        'HRAS', 'BRAF', 'MAPK3', 'JAK1', 'JAK2', 'SOCS1', 'SOCS3', 'IRF1', 'A1BG', 'A2M'
    ]
    
    # 如果需要的基因数量超过提供的，使用 GO 数据库中的前 N 个基因
    if n_genes <= len(common_genes):
        gene_names = common_genes[:n_genes]
    else:
        # 如果需要的基因更多，尝试从 GO 数据库加载更多
        try:
            import os
            import pickle
            cache_dir = os.path.expanduser('~/.cache/perturblab')
            gene2go_path = os.path.join(cache_dir, 'gene2go_all.pkl')
            if os.path.exists(gene2go_path):
                with open(gene2go_path, 'rb') as f:
                    gene2go = pickle.load(f)
                # 使用前 n_genes 个基因
                all_genes = list(gene2go.keys())[:n_genes]
                gene_names = all_genes
            else:
                # 回退：重复使用提供的基因
                gene_names = (common_genes * ((n_genes // len(common_genes)) + 1))[:n_genes]
        except Exception as e:
            print(f"警告：无法加载更多基因，使用重复的基因名称: {e}")
            gene_names = (common_genes * ((n_genes // len(common_genes)) + 1))[:n_genes]
    
    # 创建随机表达矩阵
    np.random.seed(42)
    X = np.random.lognormal(mean=2, sigma=1, size=(n_cells, n_genes)).astype(np.float32)
    
    # 创建扰动信息（使用真实的基因名称）
    conditions = []
    for i in range(n_cells):
        if i < n_cells // 3:
            conditions.append('ctrl')
        elif i < 2 * n_cells // 3:
            # 单个基因扰动 - 使用第一个真实基因
            conditions.append(gene_names[0] if len(gene_names) > 0 else 'ACTB')
        else:
            # 组合扰动 - 使用前两个真实基因
            if len(gene_names) >= 2:
                conditions.append(f'{gene_names[0]}+{gene_names[1]}')
            else:
                conditions.append('ACTB+GAPDH')
    
    # 创建 AnnData
    adata = ad.AnnData(X)
    adata.var_names = gene_names
    adata.obs['perturbation'] = conditions
    adata.obs['cell_type'] = 'test_cell'
    
    # 创建 PerturbationData
    dataset = PerturbationData(
        adata=adata,
        perturb_col='perturbation',
        control_tag='ctrl'
    )
    
    print(f"✓ 创建了 {n_cells} 个细胞，{n_genes} 个基因")
    print(f"✓ 扰动类型: {set(conditions)}")
    
    return dataset

def test_model_initialization(dataset):
    """测试模型初始化"""
    print("\n" + "="*50)
    print("测试 1: 模型初始化")
    print("="*50)
    
    config = GearsConfig(
        hidden_size=32,
        num_go_gnn_layers=1,
        num_gene_gnn_layers=1,
        decoder_hidden_size=16,
    )
    
    # 使用 init_from_dataset 初始化模型
    model = GearsModel.init_from_dataset(
        dataset=dataset,
        config=config,
        device='cpu',  # 使用 CPU 以便测试
    )
    
    print(f"✓ 模型初始化成功")
    print(f"  - 基因数量: {len(model.gene_list)}")
    print(f"  - 扰动数量: {len(model.pert_list)}")
    print(f"  - 设备: {model.device}")
    
    return model

def test_forward(model, dataset):
    """测试前向传播"""
    print("\n" + "="*50)
    print("测试 2: 前向传播")
    print("="*50)
    
    # 准备数据
    dataset.set_gears_format(fallback_cell_type='test')
    dataset.compute_de_genes(n_top_genes=10)
    dataset.pair_cells()
    
    # 如果没有 split，先进行 split
    if 'split' not in dataset.adata.obs.columns:
        dataset.split_data()
    
    # 获取 dataloader
    loader = model.get_dataloader(dataset, batch_size=16, split='train', shuffle=False)
    
    # 测试前向传播
    model.gears_model.eval()
    batch = next(iter(loader))
    batch = batch.to(model.device)
    
    output = model.forward(batch)
    
    print(f"✓ 前向传播成功")
    print(f"  - 预测形状: {output['pred'].shape}")
    if 'logvar' in output:
        print(f"  - 不确定性形状: {output['logvar'].shape}")
    
    return output

def test_compute_loss(model, dataset):
    """测试损失计算"""
    print("\n" + "="*50)
    print("测试 3: 损失计算")
    print("="*50)
    
    # 准备数据
    loader = model.get_dataloader(dataset, batch_size=16, split='train', shuffle=False)
    
    model.gears_model.train()
    batch = next(iter(loader))
    batch = batch.to(model.device)
    
    # 计算损失
    loss_dict = model.compute_loss(batch)
    
    print(f"✓ 损失计算成功")
    print(f"  - 损失值: {loss_dict['loss'].item():.4f}")
    print(f"  - 损失类型: {type(loss_dict['loss'])}")
    
    return loss_dict

def test_predict_perturbation(model, dataset):
    """测试扰动预测"""
    print("\n" + "="*50)
    print("测试 4: 扰动预测")
    print("="*50)
    
    # 准备数据
    dataset.set_gears_format(fallback_cell_type='test')
    if 'rank_genes_groups_cov_all' not in dataset.adata.uns:
        dataset.compute_de_genes(n_top_genes=10)
    if 'ctrl_indices' not in dataset.adata.obsm:
        dataset.pair_cells()
    
    # 预测
    results = model.predict_perturbation(
        dataset=dataset,
        batch_size=16,
        split='test',
        return_numpy=True
    )
    
    print(f"✓ 预测成功")
    print(f"  - 预测形状: {results['pred'].shape}")
    print(f"  - 扰动类别数量: {len(results['pert_cat'])}")
    if 'truth' in results:
        print(f"  - 真实值形状: {results['truth'].shape}")
    
    return results

def test_save_and_load(model, save_path='./test_gears_model'):
    """测试模型保存和加载"""
    print("\n" + "="*50)
    print("测试 5: 模型保存和加载")
    print("="*50)
    
    import os
    import shutil
    
    # 清理旧文件
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    # 保存模型
    model.save(save_path)
    print(f"✓ 模型已保存到: {save_path}")
    
    # 检查保存的文件
    files = os.listdir(save_path)
    print(f"  - 保存的文件: {files}")
    
    # 加载模型（需要提供 dataset 来重建 graph）
    # 注意：实际使用时，graph 已经保存在文件中，这里只是为了演示
    print("  - 注意：加载模型需要提供 dataset 或 graph")
    
    return save_path

def main():
    """主测试函数"""
    print("="*50)
    print("GEARS Model 测试")
    print("="*50)
    
    try:
        # 1. 创建测试数据
        dataset = create_test_data(n_genes=50, n_cells=100)
        
        # 2. 初始化模型
        model = test_model_initialization(dataset)
        
        # 3. 测试前向传播
        output = test_forward(model, dataset)
        
        # 4. 测试损失计算
        loss_dict = test_compute_loss(model, dataset)
        
        # 5. 测试预测
        results = test_predict_perturbation(model, dataset)
        
        # 6. 测试保存（可选，需要 graph）
        # save_path = test_save_and_load(model)
        
        print("\n" + "="*50)
        print("✓ 所有测试通过！")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
