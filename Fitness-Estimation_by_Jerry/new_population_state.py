import numpy as np
from typing import Dict, Any
from collections import Counter
from utils.constants import *
import itertools

class Gene:
    """等位基因"""
    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Gene name must be a string")
        if name.strip() == "":
            raise ValueError("Gene name cannot be empty")
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Gene) and self.name == other.name

    def __hash__(self):
        # Keys of dicts must be hashable
        return hash(self.name)


class Locus:
    """基因座，可以包含多个等位基因"""
    def __init__(self, name: str, alleles: list | tuple):
        if not isinstance(name, str):
            raise TypeError("Locus name must be a string")
        if not isinstance(alleles, (list, tuple)):
            raise TypeError("Alleles must be a list or tuple")
        
        self.name = name
        self.alleles = [Gene(a) if not isinstance(a, Gene) else a for a in alleles]

    def __repr__(self):
        return f"Locus({self.name}, alleles={[a.name for a in self.alleles]})"

    def __eq__(self, other):
        return isinstance(other, Locus) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Genome:
    """一个基因型：每个 locus 上有两个等位基因（别名 `Genotype`、`Zygote`）"""
    def __init__(self, loci_to_alleles: Dict[Locus, tuple[Gene, Gene]]):
        # 例如 {LocusA: (Gene("A"), Gene("a")), LocusB: (Gene("B"), Gene("b"))}
        if not isinstance(loci_to_alleles, dict):
            raise TypeError("Genome must be initialized with a dict {Locus: (Gene, Gene)}")

        for locus, alleles in loci_to_alleles.items():
            if not isinstance(locus, Locus):
                raise TypeError(f"Key {locus} must be a Locus")
            if not isinstance(alleles, (tuple, list)) or len(alleles) != 2:
                raise ValueError(f"Value for {locus} must be a tuple/list of length 2")
            if not all(isinstance(a, Gene) for a in alleles):
                raise TypeError(f"Alleles for {locus} must be Gene objects")

            valid_names = {a.name for a in locus.alleles}
            for a in alleles:
                if a.name not in valid_names:
                    raise ValueError(f"Allele {a} not valid for locus {locus}")

        self.loci_to_alleles = dict(loci_to_alleles)

    @property
    def gametes(self):
        """返回所有可能配子及其频率的字典 {Gamete: freq}，每个locus上等位基因频率可不同"""
        per_locus: list[list[tuple[Gene, float]]] = []
        for locus, (g1, g2) in self.loci_to_alleles.items():
            if g1 == g2:
                # 同型合子，该等位基因100%
                per_locus.append([(g1, 1.0)])
            else:
                # 杂合子，两种等位基因各50%
                per_locus.append([(g1, 0.5), (g2, 0.5)])

        gamete_freqs: Dict[Gamete, float] = {}
        for combo in itertools.product(*per_locus):
            genes, probs = zip(*combo)
            gamete = Gamete(dict(zip(self.loci_to_alleles.keys(), genes)))
            prob = 1.0
            for p in probs:
                prob *= p
            gamete_freqs[gamete] = prob

        assert np.isclose(sum(gamete_freqs.values()), 1.0), "Gamete frequencies do not sum to 1"
        return gamete_freqs

    def __repr__(self):
        return "|".join("".join(sorted([g.name for g in pair]))
                        for pair in self.loci_to_alleles.values())

    def __eq__(self, other):
        if not isinstance(other, Genome):
            return False
        return all(
            sorted([g.name for g in self.loci_to_alleles[locus]]) ==
            sorted([g.name for g in other.loci_to_alleles[locus]])
            for locus in self.loci_to_alleles
        )

    def __hash__(self):
        return hash(tuple(
            (locus.name, tuple(sorted(g.name for g in alleles)))
            for locus, alleles in sorted(self.loci_to_alleles.items(), key=lambda x: x[0].name)
        ))

# Aliases for class Genome
Zygote = Genotype = Genome


class Gamete:
    """配子：每个 locus 只有一个等位基因 （别名 `Haplotype`、`HaploidGenome`）"""
    def __init__(self, loci_to_gene: Dict):
        # {LocusA: Gene("A"), LocusB: Gene("b")}
        self.loci_to_gene = loci_to_gene

    def __repr__(self):
        return "|".join(g.name for g in self.loci_to_gene.values())

    def __eq__(self, other):
        return isinstance(other, Gamete) and self.loci_to_gene == other.loci_to_gene

    def __hash__(self):
        return hash(tuple((locus, g.name) for locus, g in self.loci_to_gene.items()))

    def combine(self, other: 'Gamete') -> Genome:
        """与另一个配子结合形成合子(Genome)"""
        loci = {}
        for locus in self.loci_to_gene:
            g1 = self.loci_to_gene[locus]
            g2 = other.loci_to_gene[locus]
            loci[locus] = (g1, g2)
        return Genome(loci)
    
# Aliases for class Gamete
Haplotype = HaploidGenome = Gamete


class GenomeSpace:
    """基因型全集（别名 `GenotypeSpace`）"""
    def __init__(self, loci: list[Locus]):
        if not isinstance(loci, (list, tuple)):
            raise TypeError("loci must be a list/tuple of Locus")
        if not all(isinstance(l, Locus) for l in loci):
            raise TypeError("All items must be Locus objects")
        self.loci = list(loci)
        self.genomes = self._generate_all_genomes()

    def _generate_all_genomes(self):
        """枚举所有合法的 Genome"""
        per_locus = []
        for locus in self.loci:
            pairs = [(a, b) for i, a in enumerate(locus.alleles)
                            for b in locus.alleles[i:]]  # 无序组合
            per_locus.append(pairs)

        genomes = []
        for combo in itertools.product(*per_locus):
            loci_to_alleles = {locus: alleles for locus, alleles in zip(self.loci, combo)}
            genomes.append(Genome(loci_to_alleles))
        return genomes
    
    def get_genome_by_repr(self, repr_str: str) -> Genome:
        for genome in self.genomes:
            if repr(genome) == repr_str:
                return genome
        raise ValueError(f"No genome found with representation: {repr_str}")

    def __iter__(self):
        return iter(self.genomes)

    def __len__(self):
        return len(self.genomes)

    def __repr__(self):
        return f"GenomeSpace({len(self.genomes)} genotypes)"

# Alias for class GenomeSpace
GenotypeSpace = GenomeSpace


class Population:
    """种群：在给定 GenomeSpace 上的分布"""
    def __init__(
        self,
        genome_space: GenomeSpace,
        genome_counts: Dict[str, dict[Genome, float]]
    ):
        """
        Args:
            genome_space: GenomeSpace 对象
            genome_counts: Dict, 形如 {"females": {<Genome>: count}, "males": {<Genome>: count}}
        """
        if not isinstance(genome_space, GenomeSpace):
            raise TypeError("Population must be tied to a GenomeSpace")
        self.genome_space = genome_space

        # 初始化分布，全集里都要出现
        self.counts: Dict[str, dict[Genome, float]] = {
            "females": {g: 0.0 for g in genome_space},
            "males": {g: 0.0 for g in genome_space}
        }
        # print("self.counts", list((id(ge), hash(ge), ge) for ge in self.counts))
        # print("genome_counts", list((id(ge), hash(ge), ge) for ge in genome_counts))

        for sex, genome_counter in genome_counts.items():
            if sex not in self.counts:
                raise ValueError(f"Unknown sex: {sex}")
            for genome, count in genome_counter.items():
                if genome not in self.counts[sex]:
                    raise ValueError(f"Genome {genome} (id {id(genome)}, hash {hash(genome)}) not in GenomeSpace")
                if count < 0:
                    raise ValueError("Counts must be non-negative")
                self.counts[sex][genome] = count

    @property
    def total(self):
        return sum(sum(sex_counts.values()) for sex_counts in self.counts.values())
    
    @property
    def total_females(self):
        return sum(self.counts["females"].values())

    @property
    def total_males(self):
        return sum(self.counts["males"].values())

    def frequencies(self):
        """返回当前种群中各基因型的频率分布，按性别"""
        tf, tm = self.total_females, self.total_males
        freqs = {
            "females": {g: (count / tf if tf > 0 else 0.0) for g, count in self.counts["females"].items()},
            "males": {g: (count / tm if tm > 0 else 0.0) for g, count in self.counts["males"].items()}
        }
        return freqs

    def _gamete_distribution(self, sex: str = "females") -> Dict[Gamete, float]:
        """统计整个种群产生的 gamete 分布，按性别区分"""
        if sex not in self.counts:
            raise ValueError(f"Unknown sex: {sex}")
        gamete_freqs = {}
        pool = self.counts[sex]
        for genome, count in pool.items():
            gametes = genome.gametes
            for g in gametes:
                gamete_freqs[g] = gamete_freqs.get(g, 0) + count * gametes[g]
        # Normalize
        total_g = sum(gamete_freqs.values())
        if total_g > 0:
            for g in gamete_freqs:
                gamete_freqs[g] /= total_g
        return gamete_freqs
    
    @property
    def sperm_pool(self) -> Dict[Gamete, float]:
        return self._gamete_distribution(sex="males")
    
    @property
    def egg_pool(self) -> Dict[Gamete, float]:
        return self._gamete_distribution(sex="females")

    # def next_generation(self):
    #     """基于随机配子结合，产生下一代种群"""
    #     gamete_dist = self.gamete_distribution()
    #     total_g = sum(gamete_dist.values())
    #     new_counts = Counter({g: 0 for g in self.genome_space})

    #     for g1, c1 in gamete_dist.items():
    #         for g2, c2 in gamete_dist.items():
    #             z = g1.combine(g2)
    #             print(f"g1: {g1}, g2: {g2} => z: {z}")
    #             new_counts[z] += (c1 / total_g) * (c2 / total_g) * self.total()

    #     print("new_counts", list((id(ge), hash(ge), ge) for ge in new_counts))

    #     return Population(self.genome_space, new_counts)

    def next_generation(self) -> 'Population':
        """
        基于配子结合，产生下一代种群
        """
        egg_dist = self.egg_pool
        sperm_dist = self.sperm_pool

        new_counts = {
            "females": {g: 0.0 for g in self.genome_space},
            "males": {g: 0.0 for g in self.genome_space}
        }

        for egg, egg_freq in egg_dist.items():
            for sperm, sperm_freq in sperm_dist.items():
                zygote = egg.combine(sperm)
                # Assuming zygote is a valid Genome object
                new_counts["females"][zygote] += egg_freq * sperm_freq
                new_counts["males"][zygote] += egg_freq * sperm_freq
        
        # TODO: Scale to total population size

        return Population(self.genome_space, new_counts)
    
# ————————

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import itertools

class VectorizedEvolutionModel:
    """
    向量化的进化模型，使用三个等位基因(D,W,R)系统
    已修改以匹配 AgeStructuredPopulation 的行为
    """
    
    # 三个等位基因的所有组合
    GENOTYPES = ['DD', 'DW', 'DR', 'WW', 'WR', 'RR']
    
    def __init__(
        self,
        initial_female_population: np.ndarray,
        initial_male_population: np.ndarray,
        initial_sperm_composition: np.ndarray,
        genotype_params: Dict[str, Dict[str, float]],
        # 年龄结构参数（匹配 AgeStructuredPopulation）
        n_ages: int = 8,
        female_adult_ages: List[int] = None,
        male_adult_ages: List[int] = None,
        female_survival_rates: List[float] = None,
        male_survival_rates: List[float] = None,
        # 驱动相关参数
        male_drive_conversion_rate: float = 0.0,
        female_drive_conversion_rate: float = 0.0,
        male_resistance_conversion_rate: float = 0.0,
        female_resistance_conversion_rate: float = 0.0,
        # 繁殖参数
        remate_chance: float = 0.05,
        offspring_per_female: int = 100,
        # 密度依赖参数
        juvenile_growth_mode: int = 1,
        recruitment_size: int = 12,
        expected_num_adult_females: float = 21.0,
        # 逻辑斯蒂
        low_density_growth_rate: float = 6.0,
        relative_competition_ability: float = 5.0,  # age-1 vs age-0 的相对竞争能力
    ):
        """
        初始化向量化进化模型
        
        Args:
            initial_female_population: 初始雌性种群 [n_genotypes, n_ages]
            initial_male_population: 初始雄性种群 [n_genotypes, n_ages]
            initial_sperm_composition: 初始精子组成 [n_ages, n_genotypes, n_genotypes]
            genotype_params: 基因型参数字典 (viability, fecundity)
            n_ages: 年龄组数 (默认8，匹配蚊子模型)
            female_adult_ages: 雌性成年年龄列表 (默认 [2,3,4,5,6,7])
            male_adult_ages: 雄性成年年龄列表 (默认 [2,3,4])
            female_survival_rates: 雌性各年龄存活率 (长度=n_ages)
            male_survival_rates: 雄性各年龄存活率 (长度=n_ages，雄性较短命)
            male_drive_conversion_rate: 雄性驱动转换率
            female_drive_conversion_rate: 雌性驱动转换率
            male_resistance_conversion_rate: 雄性抗性转换率
            female_resistance_conversion_rate: 雌性抗性转换率
            remate_chance: 重配概率 (对应 sperm_displacement_rate)
            offspring_per_female: 每雌产卵数
            juvenile_growth_mode: 幼虫生长模式 (1=密度依赖)
            recruitment_size: 每代招募到成年的数量上限
            expected_num_adult_females: 期望成年雌性数量
        """
        self.n_genotypes = len(self.GENOTYPES)
        self.n_ages = n_ages
        
        # 设置默认的年龄结构参数（匹配 AgeStructuredPopulation）
        self.female_adult_ages = female_adult_ages if female_adult_ages is not None else [2, 3, 4, 5, 6, 7]
        self.male_adult_ages = male_adult_ages if male_adult_ages is not None else [2, 3, 4]
        self.new_adult_age = min(self.female_adult_ages)  # 新成虫的年龄 = 2
        
        # 年龄依赖的存活率（匹配 AgeStructuredPopulation）
        self.female_survival_rates = np.array(female_survival_rates if female_survival_rates is not None 
                                               else [1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0.0])
        self.male_survival_rates = np.array(male_survival_rates if male_survival_rates is not None 
                                             else [1.0, 1.0, 2/3, 1/2, 0.0, 0.0, 0.0, 0.0])
        
        # 存储驱动参数
        self.male_drive_rate = male_drive_conversion_rate
        self.female_drive_rate = female_drive_conversion_rate
        self.male_resistance_rate = male_resistance_conversion_rate
        self.female_resistance_rate = female_resistance_conversion_rate
        self.remate_chance = remate_chance
        self.offspring_per_female = offspring_per_female
        
        # 密度依赖参数
        self.juvenile_growth_mode = juvenile_growth_mode
        self.recruitment_size = recruitment_size
        self.expected_num_adult_females = expected_num_adult_females
        # 逻辑斯蒂竞争参数
        self.low_density_growth_rate = low_density_growth_rate
        self.relative_competition_ability = relative_competition_ability  # age-1 vs age-0
        
        # 基因型参数字典
        self.genotype_params = genotype_params
        
        # 验证输入
        self._validate_inputs(initial_female_population, initial_male_population, initial_sperm_composition)
        
        # 初始化种群状态（直接使用数量，不是频率）
        self.female_population = initial_female_population.copy().astype(float)
        self.male_population = initial_male_population.copy().astype(float)
        # 精子存储现在是 [n_ages, n_genotypes, n_genotypes]
        self.sperm_composition = initial_sperm_composition.copy()
        
        # 预计算遗传矩阵
        self._precompute_genetic_matrices()
        
        # 存储历史
        self.history = {
            'female_population': [self.female_population.copy()],
            'male_population': [self.male_population.copy()],
            'sperm_composition': [self.sperm_composition.copy()]
        }
        
    def _validate_inputs(self, female_pop, male_pop, sperm_comp):
        """验证输入数据的形状"""
        assert female_pop.shape == (self.n_genotypes, self.n_ages), \
            f"female_pop shape should be ({self.n_genotypes}, {self.n_ages}), got {female_pop.shape}"
        assert male_pop.shape == (self.n_genotypes, self.n_ages), \
            f"male_pop shape should be ({self.n_genotypes}, {self.n_ages}), got {male_pop.shape}"
        # 精子存储现在是 [n_ages, n_genotypes, n_genotypes]
        assert sperm_comp.shape == (self.n_ages, self.n_genotypes, self.n_genotypes), \
            f"sperm_comp shape should be ({self.n_ages}, {self.n_genotypes}, {self.n_genotypes}), got {sperm_comp.shape}"
    
    def _precompute_genetic_matrices(self):
        """预计算遗传转换矩阵"""
        # 等位基因索引映射
        self.allele_to_idx = {'D': 0, 'W': 1, 'R': 2}
        self.idx_to_allele = {0: 'D', 1: 'W', 2: 'R'}
        
        # 预计算配子产生概率
        self.gamete_probs_male = self._compute_gamete_probabilities('M')
        self.gamete_probs_female = self._compute_gamete_probabilities('F')
        
        # 预计算后代概率张量
        self.offspring_probs = self._compute_offspring_probability_tensor()
    
    def _compute_gamete_probabilities(self, sex: str) -> np.ndarray:
        """计算配子产生概率矩阵 [6 genotypes × 3 alleles]"""
        gamete_probs = np.zeros((self.n_genotypes, 3))
        
        for i, genotype in enumerate(self.GENOTYPES):
            allele1, allele2 = genotype[0], genotype[1]
            
            # 基本孟德尔遗传
            if allele1 == allele2:
                base_probs = {allele1: 1.0}
            else:
                base_probs = {allele1: 0.5, allele2: 0.5}
            
            # 应用转换率
            converted_probs = self._apply_conversion_rates(base_probs, sex, genotype)
            
            # 填充矩阵
            for allele, prob in converted_probs.items():
                gamete_probs[i, self.allele_to_idx[allele]] = prob
        
        return gamete_probs
    
    def _apply_conversion_rates(self, probs: Dict[str, float], sex: str, genotype: str) -> Dict[str, float]:
        """在配子形成过程中应用驱动和抗性转换"""
        if sex == 'M':
            drive_rate = self.male_drive_rate
            resistance_rate = self.male_resistance_rate
        else:
            drive_rate = self.female_drive_rate
            resistance_rate = self.female_resistance_rate
        
        # 只对杂合子DW应用转换
        if genotype == 'DW':
            w_to_d = probs['W'] * drive_rate            
            w_to_r = probs['W'] * (1 - drive_rate) * resistance_rate
            final_w = probs['W'] - w_to_d - w_to_r
            
            probs['W'] = final_w
            probs['D'] = probs.get('D', 0.0) + w_to_d
            probs['R'] = probs.get('R', 0.0) + w_to_r
        
        # 确保概率和为1
        total = sum(probs.values())
        if total > 0:
            for allele in probs:
                probs[allele] /= total
        
        return probs
    
    def _compute_offspring_probability_tensor(self) -> np.ndarray:
        """计算后代概率张量 [雌性基因型 × 雄性基因型 × 后代基因型]"""
        offspring_probs = np.zeros((self.n_genotypes, self.n_genotypes, self.n_genotypes))
        
        # 基因型组合映射
        genotype_map = {
            'DD': 0, 'DW': 1, 'DR': 2, 
            'WD': 1, 'WW': 3, 'WR': 4, 
            'RD': 2, 'RW': 4, 'RR': 5
        }
        
        for i in range(self.n_genotypes):
            for j in range(self.n_genotypes):
                female_gametes = self.gamete_probs_female[i]
                male_gametes = self.gamete_probs_male[j]
                
                for f_allele_idx, f_prob in enumerate(female_gametes):
                    for m_allele_idx, m_prob in enumerate(male_gametes):
                        f_allele = self.idx_to_allele[f_allele_idx]
                        m_allele = self.idx_to_allele[m_allele_idx]
                        
                        # 排序等位基因
                        allele_pair = sorted([f_allele, m_allele])
                        offspring_gt = allele_pair[0] + allele_pair[1]
                        
                        offspring_idx = genotype_map[offspring_gt]
                        offspring_probs[i, j, offspring_idx] += f_prob * m_prob
        
        return offspring_probs
    
    def update_genotype_param(self, genotype: str, param_type: str, value: float):
        """更新基因型参数"""
        if genotype not in self.genotype_params:
            self.genotype_params[genotype] = {}
        self.genotype_params[genotype][param_type] = value
    
    def get_genotype_param(self, genotype: str, param_type: str) -> float:
        """获取基因型参数"""
        return self.genotype_params.get(genotype, {}).get(param_type, 1.0)
    
    def vectorized_sexual_selection(self) -> np.ndarray:
        """向量化性选择计算，返回交配概率矩阵"""
        # 获取成年雄性频率
        male_adult_freqs = self.get_adult_frequencies('M')
        
        # 构建交配偏好矩阵（这里简化，假设没有特定偏好）
        mating_matrix = np.outer(np.ones(self.n_genotypes), male_adult_freqs)
        
        # 归一化每行
        row_sums = mating_matrix.sum(axis=1, keepdims=True)
        mating_matrix = np.where(row_sums > 0, mating_matrix / row_sums, 1.0 / self.n_genotypes)
        
        return mating_matrix
    
    def update_sperm_storage(self, mating_matrix: np.ndarray):
        """
        更新精子存储，考虑重配概率
        精子存储按年龄区分：sperm_composition[age, female_gt, sperm_gt]
        """
        for age in self.female_adult_ages:
            for i in range(self.n_genotypes):
                # 获取该年龄和基因型的雌性数量
                female_count = self.female_population[i, age]
                
                if female_count > 0:
                    # 当前精子组成
                    current_sperm = self.sperm_composition[age, i, :]
                    # 新选择的精子（根据交配矩阵）
                    new_sperm = mating_matrix[i, :]
                    
                    # 应用重配概率（sperm_displacement_rate）
                    if age == self.new_adult_age:
                        # 新成年雌性，首次交配
                        updated_sperm = new_sperm.copy()
                    else:
                        # 已交配雌性，按重配概率更新
                        updated_sperm = (1 - self.remate_chance) * current_sperm + self.remate_chance * new_sperm
                    
                    # 归一化
                    if updated_sperm.sum() > 0:
                        updated_sperm /= updated_sperm.sum()
                    
                    self.sperm_composition[age, i, :] = updated_sperm
    
    def vectorized_offspring_production(self) -> np.ndarray:
        """向量化后代产生计算"""
        offspring_counts = np.zeros(self.n_genotypes)
        
        # 获取成年雌性数量和繁殖适合度
        fecundity_female = np.array([self.get_genotype_param(gt, 'fecundity') for gt in self.GENOTYPES])
        fecundity_male = np.array([self.get_genotype_param(gt, 'fecundity') for gt in self.GENOTYPES])
        
        # 遍历每个成年雌性年龄
        for age in self.female_adult_ages:
            for i, female_gt in enumerate(self.GENOTYPES):
                female_count = self.female_population[i, age]
                
                if female_count > 0:
                    # 获取该年龄、该基因型雌性的精子组成
                    sperm_composition = self.sperm_composition[age, i, :]
                    
                    # 计算雄性贡献（考虑繁殖适合度）
                    male_contributions = sperm_composition * fecundity_male
                    if male_contributions.sum() > 0:
                        male_contributions /= male_contributions.sum()
                    
                    # 计算后代基因型分布
                    for offspring_idx in range(self.n_genotypes):
                        prob = np.sum(self.offspring_probs[i, :, offspring_idx] * male_contributions)
                        
                        # 总后代数量
                        total_offspring = self.offspring_per_female * female_count * fecundity_female[i]
                        offspring_counts[offspring_idx] += total_offspring * prob
        
        return offspring_counts
    
    def add_offspring(self, offspring_counts: np.ndarray):
        """
        将后代添加到种群，使用逻辑斯蒂竞争模型
        匹配 test_mosquito_population.py 中的 recruit_juveniles_logistic 算法
        """
        # 平分到雌雄两性
        offspring_per_sex = offspring_counts / 2
        
        # 应用密度依赖（juvenile_growth_mode）
        if self.juvenile_growth_mode == 1:
            # 模式1：逻辑斯蒂竞争模型
            
            # 计算预期值
            expected_total_age_0 = self.expected_num_adult_females * self.offspring_per_female
            expected_total_age_1 = self.recruitment_size  # age_1_carrying_capacity
            expected_survival_rate = expected_total_age_1 / expected_total_age_0 if expected_total_age_0 > 0 else 1.0
            
            # 计算预期竞争强度
            expected_competition_strength = (
                expected_total_age_0 
                + expected_total_age_1 * self.relative_competition_ability
            )
            
            # 获取当前 age-0 和 age-1 幼虫数量
            # 注意：新后代 offspring_per_sex * 2 是即将成为 age-0 的
            total_age_0 = offspring_per_sex.sum() * 2  # 新产生的后代
            age_1_counts = self.female_population[:, 1] + self.male_population[:, 1]
            total_age_1 = age_1_counts.sum()
            
            # 计算实际竞争强度
            actual_competition_strength = (
                total_age_0 
                + total_age_1 * self.relative_competition_ability
            )
            
            # 计算竞争比率
            if expected_competition_strength > 0:
                competition_ratio = actual_competition_strength / expected_competition_strength
            else:
                competition_ratio = 1.0
            
            # 计算实际生长率（逻辑斯蒂公式）
            # actual_growth_rate = max(0, -competition_ratio * (r-1) + r)
            r = self.low_density_growth_rate
            actual_growth_rate = max(0.0, -competition_ratio * (r - 1) + r)
            
            # 计算实际存活率 = 生长率 × 预期存活率
            actual_survival_rate = actual_growth_rate * expected_survival_rate
            
            # 应用存活率到每个基因型
            offspring_per_sex = offspring_per_sex * actual_survival_rate
        
        # 添加到最年轻的年龄组（年龄0）
        for i in range(self.n_genotypes):
            self.female_population[i, 0] += offspring_per_sex[i]
            self.male_population[i, 0] += offspring_per_sex[i]
    
    def apply_survival_and_aging(self):
        """应用生存率和年龄推进"""
        # 获取生存适合度
        viability_female = np.array([self.get_genotype_param(gt, 'viability') for gt in self.GENOTYPES])
        viability_male = np.array([self.get_genotype_param(gt, 'viability') for gt in self.GENOTYPES])
        
        # 创建新的种群状态
        new_female_pop = np.zeros_like(self.female_population)
        new_male_pop = np.zeros_like(self.male_population)
        
        # 创建新的精子存储状态
        new_sperm_comp = np.zeros_like(self.sperm_composition)
        
        # 年龄推进和存活率应用
        for age in range(self.n_ages - 1):
            # 年龄依赖的基础存活率
            female_base_survival = self.female_survival_rates[age]
            male_base_survival = self.male_survival_rates[age]
            
            # 雌性：基础存活率 × 基因型特异存活率
            survivors_female = self.female_population[:, age] * female_base_survival * viability_female
            new_female_pop[:, age + 1] = survivors_female
            
            # 雄性：基础存活率 × 基因型特异存活率
            survivors_male = self.male_population[:, age] * male_base_survival * viability_male
            new_male_pop[:, age + 1] = survivors_male
            
            # 精子存储也要随年龄推进（只对成年雌性）
            if age + 1 in self.female_adult_ages and age in self.female_adult_ages:
                # 精子组成随雌性一起推进到下一年龄
                new_sperm_comp[age + 1, :, :] = self.sperm_composition[age, :, :]
        
        # 更新种群
        self.female_population = new_female_pop
        self.male_population = new_male_pop
        self.sperm_composition = new_sperm_comp
    
    def get_adult_counts(self, sex: str, genotype: str) -> float:
        """获取特定性别和基因型的成年个体数量"""
        pop = self.female_population if sex == 'F' else self.male_population
        genotype_idx = self.GENOTYPES.index(genotype)
        
        # 使用正确的成年年龄列表
        adult_ages = self.female_adult_ages if sex == 'F' else self.male_adult_ages
        return sum(pop[genotype_idx, age] for age in adult_ages)
    
    def get_adult_frequencies(self, sex: str) -> np.ndarray:
        """获取特定性别的成年个体基因型频率"""
        adult_counts = np.array([self.get_adult_counts(sex, gt) for gt in self.GENOTYPES])
        total = adult_counts.sum()
        return adult_counts / total if total > 0 else np.ones(self.n_genotypes) / self.n_genotypes
    
    def run_generation(self):
        """运行一个世代"""
        # 1. 性选择和精子存储更新
        mating_matrix = self.vectorized_sexual_selection()
        self.update_sperm_storage(mating_matrix)
        
        # 2. 繁殖和后代产生
        offspring_counts = self.vectorized_offspring_production()
        self.add_offspring(offspring_counts)
        
        # 3. 生存和年龄推进
        self.apply_survival_and_aging()
        
        # 4. 记录历史
        self.history['female_population'].append(self.female_population.copy())
        self.history['male_population'].append(self.male_population.copy())
        self.history['sperm_composition'].append(self.sperm_composition.copy())
    
    def run(self, num_generations: int = 1):
        """运行多个世代"""
        for _ in range(num_generations):
            self.run_generation()
    
    @property
    def current_female_frequencies(self) -> np.ndarray:
        """当前雌性基因型频率"""
        total_females = self.female_population.sum()
        return self.female_population.sum(axis=1) / total_females if total_females > 0 else np.zeros(self.n_genotypes)
    
    @property
    def current_male_frequencies(self) -> np.ndarray:
        """当前雄性基因型频率"""
        total_males = self.male_population.sum()
        return self.male_population.sum(axis=1) / total_males if total_males > 0 else np.zeros(self.n_genotypes)
    
    def get_allele_frequencies(self, sex: str = 'both') -> Dict[str, float]:
        """获取等位基因频率"""
        if sex == 'F':
            genotype_freqs = self.current_female_frequencies
        elif sex == 'M':
            genotype_freqs = self.current_male_frequencies
        else:
            # 合并两性
            total_pop = self.female_population.sum() + self.male_population.sum()
            if total_pop > 0:
                genotype_freqs = (self.female_population.sum(axis=1) + self.male_population.sum(axis=1)) / total_pop
            else:
                genotype_freqs = np.zeros(self.n_genotypes)
        
        # 计算等位基因频率
        allele_counts = {'D': 0, 'W': 0, 'R': 0}
        total_alleles = 0
        
        for i, genotype in enumerate(self.GENOTYPES):
            count = genotype_freqs[i] * 2  # 每个个体有2个等位基因
            for allele in genotype:
                allele_counts[allele] += count
            total_alleles += count * 2
        
        if total_alleles > 0:
            return {allele: count/total_alleles for allele, count in allele_counts.items()}
        else:
            return {'D': 1/3, 'W': 1/3, 'R': 1/3}  # 默认均匀分布


# 工具函数：创建默认基因型参数
def create_default_genotype_params() -> Dict[str, Dict[str, float]]:
    """创建默认的基因型参数字典"""
    genotypes = ['DD', 'DW', 'DR', 'WW', 'WR', 'RR']
    params = {}
    
    for gt in genotypes:
        params[gt] = {
            'viability': 1.0,  # 生存适合度
            'fecundity': 1.0   # 繁殖适合度
        }
    
    return params

# 工具函数：创建初始种群
def create_initial_population(
    n_genotypes: int = 6,
    n_ages: int = 8,
    initial_distribution: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建初始种群状态（匹配 AgeStructuredPopulation 格式）
    
    Args:
        n_genotypes: 基因型数量 (默认6: DD, DW, DR, WW, WR, RR)
        n_ages: 年龄组数 (默认8)
        initial_distribution: 初始分布字典，格式同 AgeStructuredPopulation
            {
                "female": {"WW": [0, 6, 6, 5, 4, 3, 2, 1], ...},
                "male": {"WW": [0, 6, 6, 4, 2, 0, 0, 0], "DW": [0, 0, 6, 0, 0, 0, 0, 0], ...}
            }
    
    Returns:
        female_population: [n_genotypes, n_ages]
        male_population: [n_genotypes, n_ages]
        sperm_composition: [n_ages, n_genotypes, n_genotypes]
    """
    # 基因型映射 (匹配 GENOTYPES = ['DD', 'DW', 'DR', 'WW', 'WR', 'RR'])
    genotype_map = {'DD': 0, 'DW': 1, 'DR': 2, 'WW': 3, 'WR': 4, 'RR': 5}
    
    female_population = np.zeros((n_genotypes, n_ages))
    male_population = np.zeros((n_genotypes, n_ages))
    
    if initial_distribution is not None:
        # 解析初始分布
        for genotype_str, counts in initial_distribution.get("female", {}).items():
            # 处理基因型字符串（如 "WT|WT" -> "WW", "WT|Drive" -> "DW"）
            gt_idx = _parse_genotype_string(genotype_str, genotype_map)
            if gt_idx is not None:
                for age, count in enumerate(counts):
                    if age < n_ages:
                        female_population[gt_idx, age] = count
        
        for genotype_str, counts in initial_distribution.get("male", {}).items():
            gt_idx = _parse_genotype_string(genotype_str, genotype_map)
            if gt_idx is not None:
                for age, count in enumerate(counts):
                    if age < n_ages:
                        male_population[gt_idx, age] = count
    else:
        # 默认：只有 WW 基因型，均匀分布在各年龄
        female_population[3, :] = 1.0  # WW
        male_population[3, :] = 1.0    # WW
    
    # 初始精子组成 [n_ages, n_genotypes, n_genotypes]
    # 初始化为全零（与 AgeStructuredPopulation 一致）
    # 雌性会在首次交配时获得精子
    sperm_composition = np.zeros((n_ages, n_genotypes, n_genotypes))
    
    return female_population, male_population, sperm_composition


def _parse_genotype_string(genotype_str: str, genotype_map: Dict[str, int]) -> Optional[int]:
    """
    解析基因型字符串，支持多种格式
    "WW", "DD", "DW" -> 直接映射
    "WT|WT" -> "WW"
    "WT|Drive" or "Drive|WT" -> "DW"
    """
    # 直接匹配
    if genotype_str in genotype_map:
        return genotype_map[genotype_str]
    
    # 解析 "X|Y" 格式
    if '|' in genotype_str:
        parts = genotype_str.split('|')
        if len(parts) == 2:
            # 转换等位基因名称
            allele_map = {'WT': 'W', 'Drive': 'D', 'Resistance': 'R', 
                          'W': 'W', 'D': 'D', 'R': 'R'}
            a1 = allele_map.get(parts[0], parts[0])
            a2 = allele_map.get(parts[1], parts[1])
            
            # 排序得到标准基因型
            sorted_alleles = ''.join(sorted([a1, a2]))
            if sorted_alleles in genotype_map:
                return genotype_map[sorted_alleles]
    
    return None


# 使用示例
if __name__ == "__main__":
    import pandas as pd
    
    # 创建默认参数
    genotype_params = create_default_genotype_params()
    
    # 使用与 test_mosquito_population.py 相同的初始分布
    initial_distribution = {
        "female": {
            "WT|WT": [0, 6, 6, 5, 4, 3, 2, 1],
        },
        "male": {
            "WT|WT":    [0, 6, 6, 4, 2, 0, 0, 0],
            "WT|Drive": [0, 0, 6, 0, 0, 0, 0, 0],
        },
    }
    
    # 创建初始种群
    female_pop, male_pop, sperm_comp = create_initial_population(
        n_genotypes=6,
        n_ages=8,
        initial_distribution=initial_distribution
    )
    
    # 初始化模型
    model = VectorizedEvolutionModel(
        initial_female_population=female_pop,
        initial_male_population=male_pop,
        initial_sperm_composition=sperm_comp,
        genotype_params=genotype_params,
        # 年龄结构（匹配 test_mosquito_population.py）
        n_ages=8,
        female_adult_ages=[2, 3, 4, 5, 6, 7],
        male_adult_ages=[2, 3, 4],
        female_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0.0],
        male_survival_rates=[1.0, 1.0, 2/3, 1/2, 0.0, 0.0, 0.0, 0.0],
        # 驱动参数（当前关闭，与 test_mosquito_population.py 一致）
        male_drive_conversion_rate=0.0,
        female_drive_conversion_rate=0.0,
        # 繁殖参数
        remate_chance=0.05,  # 对应 sperm_displacement_rate
        offspring_per_female=100,
        # 密度依赖
        juvenile_growth_mode=1,
        recruitment_size=12,
        expected_num_adult_females=21,
        # 逻辑斯蒂竞争参数（匹配 test_mosquito_population.py 中的 recruit_juveniles_logistic）
        low_density_growth_rate=6.0,  # 低密度时的增长率 r
        relative_competition_ability=5.0,  # age-1 vs age-0 的相对竞争能力
    )
    
    # 设置模拟代数
    num_generations = 10
    
    # 用于收集每代数据的列表
    records = []
    
    # 记录初始状态（第0代）
    allele_freqs = model.get_allele_frequencies()
    records.append({
        'generation': 0,
        'female_DD': model.current_female_frequencies[0],
        'female_DW': model.current_female_frequencies[1],
        'female_DR': model.current_female_frequencies[2],
        'female_WW': model.current_female_frequencies[3],
        'female_WR': model.current_female_frequencies[4],
        'female_RR': model.current_female_frequencies[5],
        'male_DD': model.current_male_frequencies[0],
        'male_DW': model.current_male_frequencies[1],
        'male_DR': model.current_male_frequencies[2],
        'male_WW': model.current_male_frequencies[3],
        'male_WR': model.current_male_frequencies[4],
        'male_RR': model.current_male_frequencies[5],
        'allele_D': allele_freqs['D'],
        'allele_W': allele_freqs['W'],
        'allele_R': allele_freqs['R'],
        'total_females': model.female_population.sum(),
        'total_males': model.male_population.sum(),
    })
    
    # 逐代运行并记录
    for gen in range(1, num_generations + 1):
        model.run(1)  # 运行1代
        
        allele_freqs = model.get_allele_frequencies()
        records.append({
            'generation': gen,
            'female_DD': model.current_female_frequencies[0],
            'female_DW': model.current_female_frequencies[1],
            'female_DR': model.current_female_frequencies[2],
            'female_WW': model.current_female_frequencies[3],
            'female_WR': model.current_female_frequencies[4],
            'female_RR': model.current_female_frequencies[5],
            'male_DD': model.current_male_frequencies[0],
            'male_DW': model.current_male_frequencies[1],
            'male_DR': model.current_male_frequencies[2],
            'male_WW': model.current_male_frequencies[3],
            'male_WR': model.current_male_frequencies[4],
            'male_RR': model.current_male_frequencies[5],
            'allele_D': allele_freqs['D'],
            'allele_W': allele_freqs['W'],
            'allele_R': allele_freqs['R'],
            'total_females': model.female_population.sum(),
            'total_males': model.male_population.sum(),
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(records)
    
    # 输出到CSV文件
    output_file = 'simulation_results_(new_population_state).csv'
    df.to_csv(output_file, index=False)
    
    print(f"模拟完成！结果已保存到: {output_file}")
    print("\n频率变化表：")
    print(df.to_string(index=False))