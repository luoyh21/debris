# **全球空间天体与空间天气监测设备元数据名录及数据库系统深度解析报告**

在现代空间交通管理（STM）、空间领域感知（SDA）以及空间天气（SWx）预报的宏观架构中，获取并管理底层观测设备自身的技术与物理属性（即传感器元数据），是实现多源数据融合、传感器资源动态调度及系统交叉定标的先决条件。随着在轨航天器数量的指数级增长以及深空探测活动的日益频繁，全球科学界、军方及商业机构的痛点已从单纯的“获取观测数据”转变为“构建全球异构传感器的元数据联邦”。  
本报告深度聚焦于全球天基与地基空间物体监测及空间天气监测的“设备数据库”与“传感器元数据名录”。区别于具体的探测数据，本报告详尽剖析国际上用于登记、管理、权限控制和检索这些监测网络物理坐标、技术参数、工作频段、视场角及运行状态的底层数据库系统与平台架构。

## **1\. 空间天气（SWx）监测设备元数据联邦与注册系统**

空间天气的实时监测高度依赖于全球分布的地基磁力计、电离层测高仪、太阳射电望远镜以及天基预警卫星阵列。为了对这些高度异构的设备进行标准化管理，国际气象与天文组织建立了多个高度结构化的设备元数据注册系统，推动传感器网络（Sensor Web Enablement, SWE）标准的落地。

### **1.1 WMO OSCAR：全球观测系统能力分析与评估元数据库**

世界气象组织（WMO）主导开发的观测系统能力分析与评估工具（OSCAR，Observing Systems Capability Analysis and Review Tool）是目前全球覆盖面最广、最权威的地球观测及空间天气设备元数据管理平台。作为 WMO 综合全球观测系统（WIGOS）的核心组件，OSCAR 并非存储气象数据的仓库，而是一个纯粹的“传感器与平台元数据联邦”1。该系统通过机器可读的接口，将全球需求与实际观测能力进行匹配。OSCAR 数据库在架构上被严格划分为三个相互关联的模块，其中 OSCAR/Space 与 OSCAR/Surface 直接涉及设备名录的管理。  
在天基设备层面，OSCAR/Space 构建了一个详尽的关系型元数据库，记录了自 1960 年 TIROS-I 卫星发射以来至 2040 年左右的全球卫星计划及其搭载仪器。该数据库目前登记了超过 1000 颗卫星和约 1200 台独立仪器，其中约有 400 台专门用于空间天气监测4。元数据字段集全面涵盖了仪器的光谱范围、空间分辨率、测量频率、定标精度、设计寿命、制造商及当前生命周期状态（如正常、降级、退役等）5。对于空间天气载荷，OSCAR 将其精细分类为太阳活动监测仪、太阳辐照度监测仪、空间辐射计/光谱仪、高能粒子能谱仪以及电磁场传感器5。  
OSCAR/Space 的核心技术壁垒在于其内置的“专家规则评估系统”（Rule-based expert system）。该系统并非一个静态的元数据展示板，而是包含近 2000 条内部开发并经独立验证的科学公式规则6。这些规则被解析为抽象语法树（AST）并作为嵌套集存储在数据库中，用于自动评估数据库中每一台仪器在测量特定气象或空间天气变量时的“适用性”。例如，通过比对仪器的扫描通道数和测量频率，系统会自动为该设备打出 1 到 5 的适用性评分，并将结果缓存于数据库中以加速关联查询4。在数据交换层面，OSCAR/Space 提供了 RESTful API，允许外部系统以 JSON 格式无缝提取仪器元数据记录1。为了应对庞大的并发检索请求，该系统近期完成了向 Microsoft Azure 云平台的整体迁移，利用 Azure App Service 和 Azure MySQL 架构大幅提升了底层元数据的响应速度与可用性4。  
在地基监测设施方面，OSCAR/Surface 充当了 WIGOS 地面观测台站（包括空间天气地基雷达、磁力计网络）的官方元数据注册表。每一个在此数据库中注册的物理监测站都会被分配一个全球唯一的 WIGOS 站点标识符（WSI）2。这一标识体系彻底解决了传统系统中由字母数字代码（如单纯的机场代码或国家气象局代码）带来的标识冲突与容量受限问题，使得任何第三方机构或科研院校部署的空间天气传感器都能被无歧义地纳入全球网络3。  
在底层数据结构上，OSCAR/Surface 采用 WMO 核心元数据配置文件（WCMP2），深度融合了 GeoJSON 与时空资产目录（STAC）标准，以 OGC API 格式对台站的物理属性进行编码。其元数据不仅包含了设备的三维空间坐标范围（geometry），还通过受控词表界定了设备的观测主题（properties.themes，如电离层特性、地磁扰动），并提供了直接指向观测数据的 API 端点链接7。更重要的是，结合 WMO 数据质量监控系统（WDQMS），OSCAR/Surface 的元数据能够被用于“预期与现实”的动态比对。系统实时监控来自全球各处理中心的观测接收率，将其与 OSCAR/Surface 中登记的“计划采集调度”元数据进行对比，从而精准定位未按时回传数据的故障传感器，生成全球传感器网络的实时健康拓扑图9。

### **1.2 ISWI：国际空间天气倡议的分布式传感器注册矩阵**

由联合国和平利用外层空间委员会（UN COPUOS）发起的国际空间天气倡议（ISWI），其核心机制并非集中控制传感器硬件，而是构建一个巨大的联邦化仪器注册目录，协调全球超过 100 个国家部署的近 1000 台地基传感器10。ISWI 秘书处（通过保加利亚科学院和日本九州大学维护的系统）通过标准化的注册表，将分布在全球的科研机构传感器联结为一个逻辑上的巨型天文台13。  
ISWI 的设备名录按科学目标被严格划分为 16 个以上的核心子网络阵列。这些子网络的元数据详细记录了设备的部署国、首席科学家、传感器类型及其空间物理测量目标。表1展示了 ISWI 框架下注册的部分关键仪器网络及其设备元数据特征16。

| 网络缩写 | 仪器网络全称 | 主要注册设备与传感器类型 | 部署规模与元数据科学目标 |
| :---- | :---- | :---- | :---- |
| **CALLISTO** | 复合天文低频低成本光谱仪与可移动天文台 | 射电频段光谱仪 / 异向天线阵列 | 监测45-870 MHz频段太阳射电爆发。全球已注册超70个站点，涉及218台接收机及73组仪器16 |
| **AMBER** | 非洲子午线B场教育与研究网络 | 磁力计阵列 | 部署于低纬度地区，监测电动力学及ULF脉动，评估内辐射带MeV电子群状态16 |
| **MAGDAS** | 磁数据采集系统 | 高精度磁力计 | 全球部署，提供连续地磁场扰动基准元数据，分析空间天气异常响应17 |
| **SCINDA** | 闪烁网络决策辅助系统 | 特种GPS/GNSS 接收机 | 记录赤道电离层扰动，其元数据用于评估高频通信退化与电磁信号闪烁概率16 |
| **RENOIR** | 偏远赤道夜间电离层区域天文台 | 光学成像仪、法布里-珀罗干涉仪 | 研究赤道及低纬度电离层-热层系统特征及其对中等及强地磁暴的响应16 |
| **CIDR** | 相干电离层多普勒接收机 | 多普勒无线电接收机 | 记录穿透电离层的卫星信号相移，提供电离层层析成像源数据，支撑数据同化模型16 |

### **1.3 专有地基空间天气高精度设备数据库**

除了综合性的 WIGOS 和 ISWI 框架，空间天气领域还存在基于特定物理量（如地磁、宇宙射线）的高精度设备数据库。这些数据库将底层台站坐标进行了高度统一和校准，是物理学家进行长期趋势分析的基石。  
SuperMAG 是一个集成全球逾 600 个地基磁力计的元数据和数据协作网络20。由于其本身不直接运营硬件，SuperMAG 的核心壁垒在于维护一个动态更新的、经历极其严格校准的台站名录。为了消除局部地理偏差，其元数据库不仅记录了台站的标准 IAGA 代码（如 FCC, TIK, NUR, BRW 等），更重要的是，它提供并维护了一套随时间插值校正的标准化磁坐标系（如不同年份的 AACGM 坐标系元数据）21。研究人员在调用 SuperMAG Web Service API 提取数据时，平台会根据这些元数据自动对原始磁场矢量进行基线扣除和局部异常修正23。与之类似，INTERMAGNET（国际实时磁性观测台网）不仅通过严格的两阶段专家审查机制管理 100 多个磁性观测台的元数据，还将地壳和岩石圈的三维电导率模型作为背景元数据嵌入数据库中，确保外部系统能精确分离出由空间天气（如极光电喷流）诱发的瞬态地磁扰动25。  
在宇宙射线监测领域，实时中子监测器数据库（NMDB）将全球分散的标准中子监测站整合为一个具有高时间分辨率（精确到 1 分钟）的统一接口名录28。该数据库管理着 18 个以上核心实时台站的元数据，包括设备的海拔高度、截止刚度（Cutoff Rigidity）及响应函数29。例如，斯洛伐克 Lomnický štít (LMKS) 台站（海拔 2634 米）的连续 42 年元数据配置及探头管数量变迁，都被详细记录于系统中，这些参数对于评估地面水平增强事件（GLE）及 Forbush 下降现象时的辐射剂量校准至关重要30。此外，由中国主导建设的子午工程（CMP），其设备名录构建了世界最大的地基空间环境监测网。在由东经 120 度等经度线分布的 15 个初始台站基础上，国际子午圈计划进一步整合了位于四川稻城的圆环阵太阳射电成像交变系统（DART，包含 313 面 6 米直径抛物面天线）以及三亚非相干散射雷达（SYISR-TS）等尖端设备的运行参数与空间坐标33。

### **1.4 天基空间天气载荷名录与事件关系数据库**

在天基层面，由 NASA 太阳物理系统天文台 (HSO) 及 NOAA 运营的空间天气卫星舰队，其载荷元数据被详细登记于相关的任务控制及分发数据库中（如 NCEI 存档库）34。这些数据库详细记录了自 20 世纪 70 年代至今的静止轨道及深空探测器的演进轨迹。  
以 NOAA 运营的 GOES（地球同步环境卫星）系列为例，其历代空间环境监测器（SEM）的元数据配置展示了传感器能力的代际跃升34。

* **GOES 1-7 (1975-1987)：** 登记的载荷包括高能粒子传感器 (EPS)、磁力计 (MAG) 和 X 射线传感器 (XRS)。数据库注明其 EPS 具备脉冲高度识别功能，可测量外辐射带捕获的质子、α粒子和电子通量34。  
* **GOES 8-12 (1994-2001)：** 新增高能质子与α粒子探测器 (HEPAD)，其元数据描述该设备包含两个硅探测器与切伦科夫辐射器照明的光电倍增管，专门用于探测 \>330 MeV 的极高能质子事件34。  
* **GOES 13-15 (2006-2010)：** 名录中更新为磁层电子探测器 (MAGED) 与磁层质子探测器 (MAGPD)。其元数据指出，这些传感器采用九台同心望远镜的十字形排列阵列（南北扇区 5 台，东西扇区 4 台），从而首次实现了高精度的粒子通量方向性解析34。  
* **深空与 L1 节点设备：** 包括 DSCOVR 以及即将部署于拉格朗日 L1 点的 SOLAR-1（前称 SWFO-L1，搭载紧凑型日冕仪 CCOR-1）37。这些设备的元数据（如校准常数、热控环境及失效通道标记）通过 netCDF 格式的属性头文件向学术界分发，确保数据反演的严谨性34。

为了将设备名录与观测到的异常关联起来，NASA 开发了 DONKI（空间天气通知、知识与信息数据库）40。DONKI 提供了一个综合性的 Web API，它不是存储原始信号，而是作为事件与设备间的关系型中枢。当某个太阳爆发事件发生时，科学家在 DONKI 中登记该事件，系统会自动链接识别该事件的特定仪器元数据（如 SOHO 卫星的某个传感器状态）及预报员日志，从而实现对空间天气因果链（日冕物质抛射、耀斑、地磁暴）的可搜索与可追溯管理40。

## **2\. 空间领域感知（SDA）与空间交通协调系统的设备元数据中枢**

与空间天气旨在监测广袤的自然等离子体环境不同，空间领域感知（SDA）及空间监视与追踪（SST）的核心是维持对近地空间中人类发射的航天器、失效卫星及数以万计的空间碎片的 custody（持续跟踪与监管）。随着低地球轨道（LEO）巨型星座（如 Starlink，占据超过一万颗活跃卫星）的迅速扩张，在轨追踪对象数量急剧上升至数万乃至十万级别41。这一现实使得全球的观测架构正经历从依赖单一地面大型雷达站，向多元异构传感器（雷达、光学、激光、被动射频）组网及元数据自动协同的深刻转变。

### **2.1 军民一体化的统一数据存储库（UDL）与 SSN 名录**

美国太空军（USSF）和空军研究实验室（AFRL）为解决传统传感器系统孤岛化的问题，开发了战略性的统一数据存储库（UDL）42。UDL 的定位远不止是一个接收数据的“数据湖”，它在本质上是一个庞大的**传感器元数据管理和权限控制中枢**。  
在架构设计上，UDL 吸纳来自美国传统军事资产、商业数据提供商（如 LeoLabs, Numerica）以及国际联盟网络的传感器观测记录、轨道状态向量及警报信息42。其核心功能在于设备元数据的校准与协方差验证。当非军方传感器将追踪数据推送到 UDL 时，其上传的元数据（包含仪器不确定性、系统噪声底准等）必须经过独立的基准测试以验证其自我一致性。例如，系统会调用如 TRACE（高精度轨道确定与弹道分析程序）等工具，将商业雷达的回波数据与高精度标定卫星的参考星历进行比对，动态识别传感器偏差，并据此调整协方差传播模型中的“过程噪声”参数。这一机制确保了即使是来源繁杂的商业传感器，其在统一目录中的权重也能被动态、科学地校准45。在安全性与互操作性上，UDL 利用公钥基础设施（PKI）实行基于角色的流级数据隔离，确保商业保密和国家安全数据互不干扰42；同时通过新近部署的 API 网关（API Gateway），支持战术边缘设备与中心数据库的微服务元数据实时交换43。  
支撑美国基础太空态势感知的物理设备名录是传统的空间监视网络（SSN）46。在最新的元数据库中，SSN 的核心资产分为地基和天基两部分：

* **地基传感器注册表：** 包含光学系统的 GEODSS（部署于新墨西哥、夏威夷及迪戈加西亚岛的1米口径深空监视系统，具备极其敏锐的暗目标探测能力），太空监视望远镜（SST），以及西班牙莫隆基地的 MOSS 系统46。雷达设施目录则更为庞大，记录有 Eglin 空军基地的 AN/FPS-85 相控阵雷达，部署于挪威的 GLOBUS II 27米单脉冲X波段雷达，以及旨在监控高纬度和导弹预警的 Cobra Dane (AN/FPS-108) 与升级版早期预警雷达系统 (UEWR)46。  
* **天基传感器星座目录：** 天基 SSA 设备的元数据至关重要，因为它们不受大气湍流、昼夜更替及云层遮蔽的限制50。目前的登记目录包括历史级别的 SBV 传感器，仍在运行的 SBSS（空间基空间监视系统，Block 10，使用可见光传感器），加拿大运营的 Sapphire，以及用于静止轨道邻近监测的 GSSAP 系统46。2023年发射升空的 Silent Barker 系统亦被编入高价值资产目录，该预警星座（包括 USA 346, 347, 348）部署在倾角约为 12 度的近地球同步轨道上，专门用于填补地面传感器对地月空间及高轨防御的盲区53。

### **2.2 欧洲空间监视与追踪系统（EU SST）的联合资产名录**

欧盟主导的 EU SST 联盟整合了包括法国、德国、意大利等 15 个成员国的高级别国家空间监测资产，其运行机制建立在一个分布式的联合设备元数据库之上57。目前该系统记录了 50 多个主要资产的物理与性能参数，其中包括 12 部雷达、34 台望远镜和 4 个激光测距站57。  
在 EU SST 的处理中心（由德国航天局等负责维护核心数据库），系统的“传感器规划系统（Sensor Planning System）”是基于该元数据名录自动运行的。当接收到可能发生碰撞（Conjunction）或有卫星机动的预警时，系统通过提取各底层设备的实时状态、视场角、信噪比阈值及追踪精度等元数据，自动计算最优观测几何，向分散在欧洲各地的传感器下发后续（follow-up）观测的调度指令58。这一自动化的工作流实现了异构传感器网络的协同，大幅提高了针对未知或未建档低轨碎片的定轨效率。

### **2.3 民用与商业 SDA 传感器的开放元数据架构**

随着 SDA 市场的商业化演进，政府机构逐渐开放民用空间交通管理平台，而商业公司则建立起规模庞大的私有传感器舰队数据库，并在数据交互接口上引领行业标准。

#### **2.3.1 TraCSS（空间交通协调系统）与开放数据交互**

美国商务部空间商务办公室（OSC）开发的 TraCSS 正在构建下一代民用和商业 SSA 元数据注册框架61。有别于军方系统严格的保密协议，TraCSS 秉承彻底的开放数据理念（遵循 CC0-1.0 许可协议）61。其基础信息库（TraCSSCat）不仅免费提供所有未保密空间对象的名录，更重要的是为卫星所有者/运营商（O/O）提供了一个开放的元数据上传通道63。截至最新数据，TraCSS 已注册 62 个领航用户（涵盖 11,230 多颗卫星），以及 7 个国家政府账户。运营商可以通过 TraCSS 开放的 API，上传其自身航天器的星历与预定机动计划，并采用带协方差信息的 OCM（Orbit Comprehensive Message）文件格式62。这种将“对象提供者的自我元数据”与“传感器网络的探测数据”相融合的机制，极大减轻了雷达网络在轨跟踪的计算压力。

#### **2.3.2 商业雷达与光学传感器网络名录**

* **LeoLabs（相控阵雷达联邦）：** LeoLabs 维护着世界上最大的商用低轨监测雷达网络数据库。其内部设备名录展现出极高的空间互补性，专门弥补了冷战时期国防网络遗留的赤道盲区和南半球间隙49。其数据库当前登记了部署于 7 个地理节点的 11 处相控阵雷达系统。其中包括位于阿拉斯加监控极地轨道的 PFISR（拥有 4096 个阵元）、覆盖德州及国际空间站轨道倾角的 Midland 雷达、新西兰的 S 波段高频雷达（分辨率达 2 厘米），以及位于哥斯达黎加（监控赤道上空）、葡萄牙亚速尔群岛（填补北大西洋盲区）和阿根廷的各类定制设备49。除了固定雷达，LeoLabs 名录中还最新增列了“Scout”及“Seeker”系列机动化、集装箱部署的三维搜索雷达，赋予了网络在几周内重新优化覆盖几何构型的物理弹性49。  
* **Slingshot Aerospace（全球光学传感器网）：** 在光学监测领域，Slingshot 维护着一个包含分布在全球 20 多个站点的 150 多台独立光学传感器的巨型注册表69。该名录在系统架构上明确区分了两种互补的设备类型：一是被称为 Varda 的受控万向节望远镜，具备特殊的滤光与遮光设计，能在局地日间实现对低轨物体的捕获，将有效观测窗口扩展了5倍；二是被称为 Horus 的“凝视光学栅栏”（Staring arrays），这些无需外部引导提示（uncued）的超大视场系统，负责在夜间连续无死角地记录视野内所有掠过的立方星级别以上的物体69。  
* **ExoAnalytic 与 Kratos：** 针对中高轨（MEO, GEO, XGEO）区域，ExoAnalytic Solutions 在其 EGTN 数据库中登记了遍布全球的 350 多台自主望远镜，通过高灵敏度（可达 18-21 等星等亮度）和低延迟（15-30秒）架构实现对深空天体的光度表征72。而在传统的射频（RF）监听链路域，Kratos 的监控网络填补了光学与雷达的盲区，其元数据目录登记了部署于全球的诸多 S, L, C, X, Ku 波段的定向射频天线系统。这些系统提取目标卫星下行链路的波形现象元数据（包括中心频率、载噪比、符号率、发射功率等），通过数据融合进行异常状态与控制链路拦截分析74。

## **3\. 天体测量与高精度激光定轨：元数据的极致标准化**

在科学界、业余天文界以及大地测量学领域，观测传感器往往独立于国家军事体系。为了实现精准的轨道拼接和长期演化预测，这些体系催生了对传感器位置参数精确度要求极高的公共设备目录代码系统。

### **3.1 IAU 小行星中心（MPC）天文台代码数据库**

国际天文学联合会（IAU）下属的小行星中心（MPC）维护着人类历史上最广泛的、针对光学和射电天体测量的“观测站元数据名录”75。近地天体（NEO）或彗星的高精度轨道拟合依赖于消除地面观测视差，这就要求极其精确的观测点地心相对位置信息。

* **编码机制的演变：** 历史上，MPC 使用由三位字符组成的代码体系（包括纯数字 000-699，700-999 以及字母开头代码如 A00-Z99）。通过这种编码组合，理论上可分配接近四万个物理观测节点75。随着自动化巡天项目的普及，系统进一步引入了基于 Base62 算法的进程代码（Program Code）扩展，以容纳同一天文台内成百上千的不同子系统及独立观测程序78。  
* **空间坐标与视差常数注册：** MPC 数据库摒弃了传统意义上的单纯经纬度标记，转而为每一个注册天文台分配严格计算的视差常数，即 ![][image1] 和 ![][image2]（其中 ![][image3] 为地心纬度，![][image4] 为以地球赤道半径为单位的地心距离）79。这种基于地球中心旋转椭球体（通常参考 WGS84）的笛卡尔衍生参数体系，使天体力学积分引擎无需在每次迭代中重算站心到地心矢量的非线性转换，极大提升了轨道确定的运算效率。  
* **非传统节点的覆盖：** 该元数据目录的强大之处在于其极高的包容性。它不仅包含如 000 英国格林尼治、644 美国帕洛马山（NEAT 巡天所在）、F51 夏威夷 Pan-STARRS 1 这类历史或现代巨型节点，还为天基光学载荷分配了特定代码（如 C49 对应 STEREO-A 太阳观测卫星）。更进一步，名录中还为地球和太阳系内的特定虚拟引力节点（如日地拉格朗日点 L1-L5 分配了 SE1-SE5 等代码）以及移动观测者（代码 247 和 XXX）分配了注册空间，从而确保哪怕是最前沿的空间碎片临时观测活动，也能找到标准的空间投影基准75。

### **3.2 ILRS（国际激光测距服务）台站编目体系**

相较于雷达反射面中心或光学镜筒轴线的不确定性，卫星激光测距（SLR）需要厘米乃至毫米级的基准面定位。国际激光测距服务（ILRS）负责协调全球网络操作和科学数据分析，并维护着一套极其严格的在役 SLR 测站、关闭台站及工程台站的元数据目录81。这些信息定期同步至两大国际数据中心：美国的 CDDIS 与德国的 EDC85。  
在元数据编目中，由于板块漂移和固体潮汐的影响，测站的位置被定义为一个具有速度分量的动态函数，而非静态坐标。ILRS 数据库为每个台站分配了多维度的追踪标识：4位物理点位代码（Monument Code）、8位 CDDIS SOD 代码，以及被国际地球自转服务（IERS）统一认证的 DOMES 编号。此外，名录还详细标识了测站的硬件追踪能力——哪些站点仅支持观测低轨/中轨目标（如 LAGEOS），哪些站点具有极高能效、能够支持地月距离级别的月球激光测距（LLR）。全球仅有少数站点具备 LLR 元数据认证，其中包括贡献了绝大多数测距历史数据的法国 Grasse MeO 站、意大利 MLRO 站、美国新墨西哥州的 APOLLO 站以及德国 WLRS 站83。表2列出了 ILRS 系统中部分具有代表性的全球测站元数据注册信息。

| 物理点位 (Monument) | 站代码 (Code) | 台站名称与隶属国家 | CDDIS SOD | IERS DOMES 编号 | IGS/IVS 联合元数据认证 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **7090** | YARL | Yarragadee, 澳大利亚 | 70900513 | 50107M001 | 南半球关键节点，涵盖全网络交叉认证83 |
| **7105** | GODL | Greenbelt, 马里兰, 美国 | 71050725 | 40451M105 | NASA Goddard 中心，系统参考零点基准83 |
| **7237** | CHAL | 长春, 中国 | 72371901 | 21611S001 | 观测效率极高的主力测站之一83 |
| **7810** | ZIML | Zimmerwald, 瑞士 | 78106801 | 14001S007 | AIUB主管，融合天体测量与光学跟踪配置83 |
| **7845** | GRSM | Grasse, 法国 (具备LLR能力) | 78457801 | 10002S002 | 负责贡献了长期月球激光测距绝大部分数据83 |

### **3.3 科学界志愿观测网与数据格式的演进**

在国家级资产之外，由俄罗斯科学院克尔德什应用数学研究所（KIAM）主导的国际科学光学网络（ISON），通过开放的志愿合作机制构建了一个独立的监控网络90。ISON 的设备元数据注册簿涵盖了分布在 17 个国家 27 个天文台的 50 多台望远镜91。该名录基于望远镜的孔径和视场（FOV）对任务进行分类。例如，位于远东的乌苏里斯克及格鲁吉亚等地的 22-25 厘米口径、大视场望远镜（如 ORI-22/ORI-25，采用 Hamilton-Newton 光学系统）被专门注册用于 GEO 轨道的系统性扫描；而较大口径设备则被调度针对高面积质量比（HAMR）微小碎片的深度跟踪90。  
值得重点关注的是，无论是庞大的军方库还是业余无线电操作员（如 HamSat 平台或 SeeSat-L 邮件列表社区92）使用的民用目录，其底层“物体跟踪元数据”格式在近期经历了历史性的重构。自 2026 年 7 月 11 日起，随着在轨被追踪物体及碎片正式突破 99,999 的物理上限，美国太空军授权的卫星目录体系不得不终结长期使用的、继承自 20 世纪 60 年代 IBM 80 列打孔卡时代的传统两行轨道根数（TLE）固定字段格式。作为替代，全新的目录系统全面启用了 Alpha-5 编码机制。同时，为了支撑更现代的编程接口环境，CelesTrak 等数据分发中心开始使用结构化的 JSON、XML 及 KVN 等格式分发最新的通用扰动（GP）星历元数据93。这些格式的演进彻底解除了传统元数据中隐含的小数点和两位数年份带来的千年虫式危机，并使得现代手持终端与天文控制软件（结合如 SGP4 轨道传播引擎的本地部署）在进行数据解算和云端同步时，能够享受到极高的稳定性与运算速率92。

## **结语**

从本质上而言，无论是追踪以每秒几公里速度穿梭于低地轨道的高反照率卫星碎片，还是预测地球磁层在几日后对特定日冕质量抛射的高能粒子冲刷响应，空间科学早已转变为一个极限尺度的分布式计算与网络协调问题。在这一背景下，庞大观测数据的价值和可信度，绝对受制于对捕捉这些数据的硬件设备自身技术轮廓的精准刻画。  
通过 WMO 的 OSCAR（气象与空间天气天地基元数据总览）、IAU 的 MPC（极度精细的天文台视差字典）、ILRS（厘米级激光参考站）、ISWI（分布于发展中国家及前沿阵地的空间天气仪），以及美国太空军构建的 UDL 和欧盟的 EU SST（军民融合的动态调度联邦池）等极其复杂的注册与服务框架，全球航天、气象与天文学界实际上已经编织出了一张覆盖全频段、全空间尺度且具备高度机器互操作性的底层感知神经网络。展望未来，随着太空商业化步伐的不断加快及空间交通协调系统（如 TraCSS）的全面落地，空间监控将更深层次地依赖于这些以云计算和高速 API 驱动的自动化“设备数据库”，在毫秒级内完成从异构元数据握手、观测资源指派到规避碰撞预警的一体化信息闭环。

#### **引用的著作**

> 1. WMO OSCAR | Observing Systems Capability Analysis and Review Tool \- Home, [https://space.oscar.wmo.int/](https://space.oscar.wmo.int/)  
> 2. WMO Initiatives \- National Weather Service, [https://www.weather.gov/datamgmt/WMO\_Initiatives](https://www.weather.gov/datamgmt/WMO_Initiatives)  
> 3. Guide to the WMO Integrated Global Observing System \- amc@namem.gov.mn, [https://amc.namem.gov.mn/wp-content/uploads/WMO/19.%201165-2024-edition\_en.pdf?\_t=1638837870](https://amc.namem.gov.mn/wp-content/uploads/WMO/19.%201165-2024-edition_en.pdf?_t=1638837870)  
> 4. Status and plans of WMO OSCAR/Space database \- The Coordination Group for Meteorological Satellites, [https://www.cgms-info.org/Agendas/GetWpFile.ashx?wid=ae34b0be-f77d-470d-9196-05e3939b268d\&aid=8f1ac495-d20f-4675-abd2-859ef2714faa](https://www.cgms-info.org/Agendas/GetWpFile.ashx?wid=ae34b0be-f77d-470d-9196-05e3939b268d&aid=8f1ac495-d20f-4675-abd2-859ef2714faa)  
> 5. List of all Instruments \- WMO OSCAR, [https://space.oscar.wmo.int/instruments](https://space.oscar.wmo.int/instruments)  
> 6. WMO OSCAR/Space \- infostreams, [https://www.infostreams.net/projects/wmo-oscar](https://www.infostreams.net/projects/wmo-oscar)  
> 7. WMO Core Metadata Profile (WCMP) Version 2, [https://wmo-im.github.io/wcmp2/standard/wcmp2-STABLE.html](https://wmo-im.github.io/wcmp2/standard/wcmp2-STABLE.html)  
> 8. WMO WIS 2.0 Discovery Metadata exchange, harvesting and search pilot: Project Report, [https://wmo-im.github.io/wis2-metadata-search/](https://wmo-im.github.io/wis2-metadata-search/)  
> 9. WMO and ECMWF launch new web tool to monitor quality of observations, [https://www.ecmwf.int/en/about/media-centre/news/2020/wmo-and-ecmwf-launch-new-web-tool-monitor-quality-observations](https://www.ecmwf.int/en/about/media-centre/news/2020/wmo-and-ecmwf-launch-new-web-tool-monitor-quality-observations)  
> 10. International Space Weather Initiative, [https://www.ion.org/publications/abstract.cfm?articleID=10888](https://www.ion.org/publications/abstract.cfm?articleID=10888)  
> 11. Space Weather \- UNOOSA, [https://www.unoosa.org/oosa/en/ourwork/topics/space-weather.html](https://www.unoosa.org/oosa/en/ourwork/topics/space-weather.html)  
> 12. International Space Weather Initiative (ISWI) \- UNOOSA, [https://www.unoosa.org/oosa/en/ourwork/psa/bssi/iswi.html](https://www.unoosa.org/oosa/en/ourwork/psa/bssi/iswi.html)  
> 13. the International Space Weather Initiative (Secretariat), [http://www.stil.bas.bg/ISWI/](http://www.stil.bas.bg/ISWI/)  
> 14. International Space Weather Initiative Workshop | (smr 3292\) (20-24 May 2019\) \- Indico, [https://indico.ictp.it/event/8682/](https://indico.ictp.it/event/8682/)  
> 15. ISWI-SECRETARIAT.ORG site MAP, [http://www.stil.bas.bg/ISWI/ISWI\_map1.html](http://www.stil.bas.bg/ISWI/ISWI_map1.html)  
> 16. (PDF) The International SpaceWeather Initiative (ISWI) \- ResearchGate, [https://www.researchgate.net/publication/226275178\_The\_International\_SpaceWeather\_Initiative\_ISWI](https://www.researchgate.net/publication/226275178_The_International_SpaceWeather_Initiative_ISWI)  
> 17. United Nations Basic Space Science Initiative: 2011 Status Report on the International Space Weather Initiative \- arXiv, [https://arxiv.org/pdf/1108.2247](https://arxiv.org/pdf/1108.2247)  
> 18. Ionosphere Monitoring and Prediction Center (IMPC), [https://elib.dlr.de/218402/2/978-981-95-1121-1\_33.pdf](https://elib.dlr.de/218402/2/978-981-95-1121-1_33.pdf)  
> 19. TOWARDS A NEXT-GENERATION eCallisto NETWORK \- University of Hertfordshire, [https://star.herts.ac.uk/cesra/presentations/Bussons.pdf](https://star.herts.ac.uk/cesra/presentations/Bussons.pdf)  
> 20. About SuperMAG, [https://supermag.jhuapl.edu/info/](https://supermag.jhuapl.edu/info/)  
> 21. Inventory \- SuperMAG, [https://supermag.jhuapl.edu/inventory/](https://supermag.jhuapl.edu/inventory/)  
> 22. Minutes of the Business Meeting of IAGA WG V-OBS \- Geomagnetic Observation, [https://iaga-vobs.org/docs/Minutes\_WG\_V\_OBS\_2017.pdf](https://iaga-vobs.org/docs/Minutes_WG_V_OBS_2017.pdf)  
> 23. Download Data \- SuperMAG, [https://supermag.jhuapl.edu/mag/](https://supermag.jhuapl.edu/mag/)  
> 24. Search for ultralight dark matter in the SuperMAG high-fidelity dataset \- CERN, [https://scoap3-prod-backend.s3.cern.ch/media/files/90778/10.1103/PhysRevD.110.115036.pdf](https://scoap3-prod-backend.s3.cern.ch/media/files/90778/10.1103/PhysRevD.110.115036.pdf)  
> 25. Special issue – Geomagnetic observatories, their data, and the application of their data \- GI, [https://gi.copernicus.org/articles/special\_issue1331.html](https://gi.copernicus.org/articles/special_issue1331.html)  
> 26. Data & Services | EPOS \- the European Plate Observing System, [https://www.epos-eu.org/tcs/geomagnetic-observations/data-services](https://www.epos-eu.org/tcs/geomagnetic-observations/data-services)  
> 27. Geomagnetic observatories \- GFZ, [https://www.gfz.de/en/section/geomagnetism/infrastructure/geomagnetic-observatories](https://www.gfz.de/en/section/geomagnetism/infrastructure/geomagnetic-observatories)  
> 28. Real-time Neutron Monitor Database \- Wikipedia, [https://en.wikipedia.org/wiki/Real-time\_Neutron\_Monitor\_Database](https://en.wikipedia.org/wiki/Real-time_Neutron_Monitor_Database)  
> 29. Neutron monitor stations worldwide. \- ResearchGate, [https://www.researchgate.net/figure/Neutron-monitor-stations-worldwide\_fig1\_222889208](https://www.researchgate.net/figure/Neutron-monitor-stations-worldwide_fig1_222889208)  
> 30. Analysis of 42 years of Cosmic Ray Measurements by the Neutron Monitor at Lomnický štít Observatory \- arXiv, [https://arxiv.org/html/2502.09627v1](https://arxiv.org/html/2502.09627v1)  
> 31. Potential use of NMDB for the real-time Observation and Specification of the near-Earth Radiation environment \- Solar Influences Data Analysis Center, [https://www.sidc.be/esww7/presentations/4.4.pdf](https://www.sidc.be/esww7/presentations/4.4.pdf)  
> 32. Can Cosmic Rays Neutron Sensors provide valuable data about space weather events? \- Meeting Organizer, [https://meetingorganizer.copernicus.org/EGU26/EGU26-11732.html?pdf](https://meetingorganizer.copernicus.org/EGU26/EGU26-11732.html?pdf)  
> 33. China completes world's largest ground-based network for space weather monitoring \- Oxford Academic, [https://academic.oup.com/nsr/article/12/6/nwae354/7817990](https://academic.oup.com/nsr/article/12/6/nwae354/7817990)  
> 34. GOES 1–15 Space Weather Instruments \- National Centers for Environmental Information, [https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments)  
> 35. Search \- NASA SVS, [https://svs.gsfc.nasa.gov/search/?search=Heliophysics+Infographic](https://svs.gsfc.nasa.gov/search/?search=Heliophysics+Infographic)  
> 36. The 2023 Senior Review of the Heliophysics System Observatory Missions | NASA Science, [https://science.nasa.gov/wp-content/uploads/2023/09/final-2023-senior-review-report-tagged.pdf](https://science.nasa.gov/wp-content/uploads/2023/09/final-2023-senior-review-report-tagged.pdf)  
> 37. SOLAR-1 Launch \- National Environmental Satellite, Data, and Information Service NESDIS, [https://www.nesdis.noaa.gov/our-satellites/currently-flying/solar-1-launch](https://www.nesdis.noaa.gov/our-satellites/currently-flying/solar-1-launch)  
> 38. List of Solar System probes \- Wikipedia, [https://en.wikipedia.org/wiki/List\_of\_Solar\_System\_probes](https://en.wikipedia.org/wiki/List_of_Solar_System_probes)  
> 39. NASA NOAA \- SPACE WEATHER PROGRAM Project \- Registry of Open Data on AWS, [https://registry.opendata.aws/nasa-noaa---space-weather-program/](https://registry.opendata.aws/nasa-noaa---space-weather-program/)  
> 40. Space Weather Database Of Notifications, Knowledge, Information (DONKI) | T2 Portal, [https://technology.nasa.gov/patent/GSC-TOPS-223](https://technology.nasa.gov/patent/GSC-TOPS-223)  
> 41. How Many Satellites Are in Orbit in 2026? — Live Count | Orbital Radar, [https://orbitalradar.com/how-many-satellites-in-orbit](https://orbitalradar.com/how-many-satellites-in-orbit)  
> 42. Commercial Capabilities in SSA Data and STM Services \- Regulations.gov, [https://downloads.regulations.gov/DOC-2019-0001-0015/attachment\_1.pdf](https://downloads.regulations.gov/DOC-2019-0001-0015/attachment_1.pdf)  
> 43. API Gateway to Boost USSF Space Superiority Through Enhanced Data Access \- NSTXL, [https://nstxl.org/ssc-api-gateway/](https://nstxl.org/ssc-api-gateway/)  
> 44. SPACE SITUATIONAL AWARENESS DOD Should Evaluate How It Can Use Commercial Data \- Government Accountability Office (GAO), [https://www.gao.gov/assets/820/819460.pdf](https://www.gao.gov/assets/820/819460.pdf)  
> 45. Establishing an Independent Data Quality Analysis Framework for UDL Published \- AMOS Conference, [https://amostech.com/TechnicalPapers/2020/SSA-SDA/Gonring.pdf](https://amostech.com/TechnicalPapers/2020/SSA-SDA/Gonring.pdf)  
> 46. United States Space Surveillance Network \- Wikipedia, [https://en.wikipedia.org/wiki/United\_States\_Space\_Surveillance\_Network](https://en.wikipedia.org/wiki/United_States_Space_Surveillance_Network)  
> 47. A SYSTEMATIC EXAMINATION OF GROUND-BASED AND SPACE-BASED APPROACHES TO OPTICAL DETECTION AND TRACKING OF SATELLITES Introduction \- 41st Space Symposium, [https://www.spacesymposium.org/wp-content/uploads/2017/10/M.Ackermann\_31st\_Space\_Symposium\_Tech\_Track\_paper.pdf](https://www.spacesymposium.org/wp-content/uploads/2017/10/M.Ackermann_31st_Space_Symposium_Tech_Track_paper.pdf)  
> 48. Performance Analysis of Sensor Systems for Space Situational Awareness, [https://www.janss.kr/archive/view\_article?pid=jass-34-303](https://www.janss.kr/archive/view_article?pid=jass-34-303)  
> 49. LeoLabs and the Business of Watching Everything \- KeepTrack Space, [https://keeptrack.space/deep-dive/leolabs](https://keeptrack.space/deep-dive/leolabs)  
> 50. Optimal Tasking and Scheduling of Satellite Constellations for Space Situational Awareness Allan Shtofenmakher Massachusetts Ins \- AMOS Conference, [https://amostech.com/TechnicalPapers/2025/Poster/Shtofenmakher.pdf](https://amostech.com/TechnicalPapers/2025/Poster/Shtofenmakher.pdf)  
> 51. SBSS (Space-Based Surveillance System) \- eoPortal, [https://www.eoportal.org/satellite-missions/sbss](https://www.eoportal.org/satellite-missions/sbss)  
> 52. Space-Based Space Surveillance \- Air & Space Forces Magazine, [https://www.airandspaceforces.com/weapons/sbss/](https://www.airandspaceforces.com/weapons/sbss/)  
> 53. [https://www.airandspaceforces.com/weapons/silent-barker/\#:\~:text=It%20was%20launched%20in%20September,at%20a%2012%2Ddegree%20inclination.](https://www.airandspaceforces.com/weapons/silent-barker/#:~:text=It%20was%20launched%20in%20September,at%20a%2012%2Ddegree%20inclination.)  
> 54. Silent Barker \- Learn Finite, [https://www.learnfinite.com/ca/silent-barker](https://www.learnfinite.com/ca/silent-barker)  
> 55. Space Force, NRO launch 'Silent Barker' space observation satellites \- C4ISRNet, [https://www.c4isrnet.com/battlefield-tech/space/2023/09/10/space-force-nro-launch-silent-barker-space-observation-satellites/](https://www.c4isrnet.com/battlefield-tech/space/2023/09/10/space-force-nro-launch-silent-barker-space-observation-satellites/)  
> 56. What is Silent Barker? \- GKToday, [https://www.gktoday.in/what-is-silent-barker/](https://www.gktoday.in/what-is-silent-barker/)  
> 57. IAC-20-E9.1-A6.8-58173 Page 1 of 5 IAC-20-E9.1-A6.8-58173 Recent Developments in the Implementation of European Space Survei \- EU SST, [https://www.eusst.eu/sites/default/files/documents/IAC2020\_E9.1-A6.8\_Paper\_EUSST.pdf](https://www.eusst.eu/sites/default/files/documents/IAC2020_E9.1-A6.8_Paper_EUSST.pdf)  
> 58. EU Space Surveillance and Tracking Service Portfolio, [https://www.eusst.eu/sites/default/files/documents/EUSST\_Service\_Portfolio.pdf](https://www.eusst.eu/sites/default/files/documents/EUSST_Service_Portfolio.pdf)  
> 59. ESA \- Space Surveillance and Tracking \- SST Segment, [https://www.esa.int/Space\_Safety/Space\_Surveillance\_and\_Tracking\_-\_SST\_Segment](https://www.esa.int/Space_Safety/Space_Surveillance_and_Tracking_-_SST_Segment)  
> 60. IAC-07-A6.I.10 A POSSIBLE WAY OF EXCHANGING FOLLOW-UP DATA Tim Flohrer Thomas Schildknecht, Reto Musci, [https://ccd.aiub.unibe.ch/publist/data/2007/artproc/TF\_IAC2007a.pdf](https://ccd.aiub.unibe.ch/publist/data/2007/artproc/TF_IAC2007a.pdf)  
> 61. TraCSS User Agreement & Data Policy \- Office of Space Commerce, [https://space.commerce.gov/traffic-coordination-system-for-space-tracss/tracss-user-agreement-data-policy-private/](https://space.commerce.gov/traffic-coordination-system-for-space-tracss/tracss-user-agreement-data-policy-private/)  
> 62. Traffic Coordination System for Space (TraCSS). \- Office of Space Commerce, [https://space.commerce.gov/traffic-coordination-system-for-space-tracss/](https://space.commerce.gov/traffic-coordination-system-for-space-tracss/)  
> 63. TraCSS Overview Presentation \- June 2026.pptx \- UNOOSA, [https://www.unoosa.org/documents/pdf/copuos/2026/TPs/Materials/PDF\_Slides\_USA\_Borowitz.pdf](https://www.unoosa.org/documents/pdf/copuos/2026/TPs/Materials/PDF_Slides_USA_Borowitz.pdf)  
> 64. Traffic Coordination System for Space (TraCSS) Data Policy, User Agreement, and User Information Collection Plan, [https://www.space.commerce.gov/wp-content/uploads/2025/03/TraCSS-Data-Policy-User-Agreement-and-Registration-Final\_03-10-2025.pdf](https://www.space.commerce.gov/wp-content/uploads/2025/03/TraCSS-Data-Policy-User-Agreement-and-Registration-Final_03-10-2025.pdf)  
> 65. About Us \- TraCSS, [https://tracss.gov/about](https://tracss.gov/about)  
> 66. LeoLabs \- Wikipedia, [https://en.wikipedia.org/wiki/LeoLabs](https://en.wikipedia.org/wiki/LeoLabs)  
> 67. LeoLabs Deploys Scout-S™, Establishing A New Class of Transportable 3D Search Radars to Enhance Battlespace Awareness for Space Operations | Morningstar, [https://www.morningstar.com/news/pr-newswire/20260610la80030/leolabs-deploys-scout-s-establishing-a-new-class-of-transportable-3d-search-radars-to-enhance-battlespace-awareness-for-space-operations](https://www.morningstar.com/news/pr-newswire/20260610la80030/leolabs-deploys-scout-s-establishing-a-new-class-of-transportable-3d-search-radars-to-enhance-battlespace-awareness-for-space-operations)  
> 68. LeoLabs Achieves Record Bookings in 2025 Fueled by Triple Digit Growth in U.S. Government Contracts, [https://leolabs.space/press/leolabs-achieves-record-bookings-in-2025-fueled-by-triple-digit-growth-in-u-s-government-contracts/](https://leolabs.space/press/leolabs-achieves-record-bookings-in-2025-fueled-by-triple-digit-growth-in-u-s-government-contracts/)  
> 69. Safeguarding the Final Frontier: Optical Sensors for Persistent Space Domain Awareness, [https://www.slingshot.space/news/why-optical](https://www.slingshot.space/news/why-optical)  
> 70. Slingshot Aerospace Expands Global Sensor Network to Create World's Largest Commercial Optical Tracking System for LEO Satellites \- Business Wire, [https://www.businesswire.com/news/home/20230412005258/en/Slingshot-Aerospace-Expands-Global-Sensor-Network-to-Create-Worlds-Largest-Commercial-Optical-Tracking-System-for-LEO-Satellites](https://www.businesswire.com/news/home/20230412005258/en/Slingshot-Aerospace-Expands-Global-Sensor-Network-to-Create-Worlds-Largest-Commercial-Optical-Tracking-System-for-LEO-Satellites)  
> 71. Why Slingshot Aerospace is one of 2024's most innovative companies, [https://www.fastcompany.com/91037723/slingshot-aerospace-most-innovative-companies-2024](https://www.fastcompany.com/91037723/slingshot-aerospace-most-innovative-companies-2024)  
> 72. Anduril moves to boost space surveillance with acquisition of ExoAnalytic Solutions, [https://www.spaceconnectonline.com.au/situational-awareness/6994-anduril-moves-to-boost-space-surveillance-with-acquisition-of-exoanalytic-solutions](https://www.spaceconnectonline.com.au/situational-awareness/6994-anduril-moves-to-boost-space-surveillance-with-acquisition-of-exoanalytic-solutions)  
> 73. Space Intelligence \- ExoAnalytic Solutions, [https://exoanalytic.com/space-intelligence/](https://exoanalytic.com/space-intelligence/)  
> 74. Towards Complete Space Domain Awareness, [https://www.kratosspace.com/-/media/k/pdf/s/si/kratos-rf-space-situational-awareness.pdf](https://www.kratosspace.com/-/media/k/pdf/s/si/kratos-rf-space-situational-awareness.pdf)  
> 75. List of observatory codes \- Grokipedia, [https://grokipedia.com/page/List\_of\_observatory\_codes](https://grokipedia.com/page/List_of_observatory_codes)  
> 76. Links \- Ageo Observatory, [https://www.astroarts.co.jp/ageo/link/index.html](https://www.astroarts.co.jp/ageo/link/index.html)  
> 77. MPC observatory codes \- Project Pluto, [https://projectpluto.com/rovers.htm](https://projectpluto.com/rovers.htm)  
> 78. Program Codes \- IAU Minor Planet Center, [https://www.minorplanetcenter.net/mpcops/documentation/program-codes/](https://www.minorplanetcenter.net/mpcops/documentation/program-codes/)  
> 79. IAU Observatory codes \- Formulaires de l'IMCCE, [https://ssp.imcce.fr/webservices/data/displayIAUObsCodes.php](https://ssp.imcce.fr/webservices/data/displayIAUObsCodes.php)  
> 80. List Of Observatory Codes \- Project Pluto, [https://www.projectpluto.com/neocp2/ObsCodesF.html](https://www.projectpluto.com/neocp2/ObsCodesF.html)  
> 81. ILRS – Satellite & Lunar Laser Ranging Service | IAG-GGOS \- Geodesy | Science, [https://geodesy.science/item/ilrs/](https://geodesy.science/item/ilrs/)  
> 82. ILRS: Current Status and Future Plans \- International Laser Ranging Service, [https://ilrs.gsfc.nasa.gov/lw21/docs/2018/papers/Session1\_Noll\_paper.pdf](https://ilrs.gsfc.nasa.gov/lw21/docs/2018/papers/Session1_Noll_paper.pdf)  
> 83. ILRS | Network | Stations | Active Stations | \- International Laser Ranging Service, [https://ilrs.gsfc.nasa.gov/network/stations/active/](https://ilrs.gsfc.nasa.gov/network/stations/active/)  
> 84. ILRS | Network | Stations | \- International Laser Ranging Service \- NASA, [https://ilrs.gsfc.nasa.gov/network/stations/index.html](https://ilrs.gsfc.nasa.gov/network/stations/index.html)  
> 85. ILRS | Data and Products | \- International Laser Ranging Service, [https://ilrs.dgfi.tum.de/data\_and\_products/index.html](https://ilrs.dgfi.tum.de/data_and_products/index.html)  
> 86. ILRS | Network | Stations | Engineering Stations | \- International Laser Ranging Service \- NASA, [https://ilrs.gsfc.nasa.gov/network/stations/engineering/index.html](https://ilrs.gsfc.nasa.gov/network/stations/engineering/index.html)  
> 87. 3.4.2 International Laser Ranging Service (ILRS), [https://www.iers.org/fileadmin/SharedDocs/Publikationen/EN/IERS/Publications/ar/ar2013/ar2013\_342.pdf](https://www.iers.org/fileadmin/SharedDocs/Publikationen/EN/IERS/Publications/ar/ar2013/ar2013_342.pdf)  
> 88. IERS Annual Report 2019, [https://www.iers.org/fileadmin/SharedDocs/Publikationen/EN/IERS/Publications/ar/ar2019/ar2019\_343.pdf](https://www.iers.org/fileadmin/SharedDocs/Publikationen/EN/IERS/Publications/ar/ar2019/ar2019_343.pdf)  
> 89. Space Surveillance System, [https://www.esa.int/esapub/bulletin/bulletin133/bul133f\_klinkrad.pdf](https://www.esa.int/esapub/bulletin/bulletin133/bul133f_klinkrad.pdf)  
> 90. Search and study of the space debris and asteroids within ISON project \- SciELO, [https://www.scielo.br/j/aabc/a/dwVndYKfWPGjvhdg6M4ZVbf/?lang=en](https://www.scielo.br/j/aabc/a/dwVndYKfWPGjvhdg6M4ZVbf/?lang=en)  
> 91. ISON SEARCH AND STUDY THE NEAR-EARTH SPACE OBJECTS \- ESA Proceedings Database |, [https://conference.sdo.esoc.esa.int/proceedings/neosst1/paper/406/NEOSST1-paper406.pdf](https://conference.sdo.esoc.esa.int/proceedings/neosst1/paper/406/NEOSST1-paper406.pdf)  
> 92. HamSat \- App Store, [https://apps.apple.com/sg/app/hamsat/id6762450814](https://apps.apple.com/sg/app/hamsat/id6762450814)  
> 93. Two-Line Element Set (TLE) \- KeepTrack Space, [https://keeptrack.space/space-terms/two-line-element-set](https://keeptrack.space/space-terms/two-line-element-set)  
> 94. Resources, [https://www.mindseyeobservatory.org/home/resources](https://www.mindseyeobservatory.org/home/resources)  
> 95. ANS-200 AMSAT News Service Weekly Bulletins, [https://www.amsat.org/ans-200-amsat-news-service-weekly-bulletins/](https://www.amsat.org/ans-200-amsat-news-service-weekly-bulletins/)  
> 96. A Full-Stack Web Architecture Approach: Real-Time Orbital Tracking and Visualization \- RSIS International, [https://rsisinternational.org/journals/ijriss/uploads/vol10-iss4-pg1140-1152-202604\_pdf.pdf](https://rsisinternational.org/journals/ijriss/uploads/vol10-iss4-pg1140-1152-202604_pdf.pdf)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAaCAYAAADv/O9kAAACr0lEQVR4Xu2XTahNURTHl1BI5KMnIZKJkhJeKTJRGJB8RBkoEwZGBmSmZPAyk5FIBlIyNTO4GCiUiZhQb0AGigwMKB/rd9fZnXXW3cc9N5PrdH7179691jn7a6299r0iHR0dHR3/FWtVv1V7oqNN7FDNCrb9qmnVimBvDTNUZ4Ntvqqn2hbsrWKRanuwkeYfVXODvVWcksE0v6x6H2ytgjS/FY3KQ9W1aBzGAtVPsTPC55WKt8o91S/VTbEJfFJtcf6jqq9iEUB8x5aYrZoqdFisGNHfME6qPhRiTN556vzzVDNdeyjrVK9U64v2BbErgZ2NLBTznSvavEP7QdHmHdqXijbw3fd3WvW9dPf7fOzaObaq3qoOiKX5fdVKsSgfcs81Jk30orNtVn2TweKxQfVFbGDPPrGMmSO2ATeq7v4Yd1R3xc7lbbExl7ln/nb9MCYRTjyTcm5p/sdLdzPSYvhMsBA649PDgrHHDUmkqno+OsQWS+EhSsfE+kGfxdK9rhJjj4F5IdVNw58bsxZ2n5S5GuxEjc6WBjudYycjcqRMyU0CGz7/LhtJqqZNmHC+BM88F7u+gAj7/jlqnPVdzjYU0mta7NeOh6gxkQjphH1vdBSkiMdUByL+TixSJ1QbnW+3WNbl+mWRPbEfJxDv7zNS9TciRZAJJ1aLVfVcwWC3KVQsYI2zH5SyOE3K4NFJx2lT0WYT3pTuPkQ2d4SWSLU/Nj/d34z5o/g+Eimlr4tdMZw/Ojsi+YoOqaq/dLbXUs0aJuOrdJxgKm4URFiseiL1UWPhpHu6v5krG0CVp9qPDGeD1OTu43zVDZyDwXeqlkeHA1/Ov0rsfcbD33Rc/nFx7TZ9vhZ2vReNYwxRphj/ExQZFh4L2zjzSPJ1YCTogDT3hW3coQ7EK7ajo2OQPwzDjJxK7obvAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAaCAYAAADrCT9ZAAACiklEQVR4Xu2WTciMURTHj1BEPt9IlIiFiIUsFCsUhURKsaIoWdnIzt7Ku1R6XysbGws7i8FCPlayEvUqsrBQYkH5+P869zZn7nw8MxlM4/nVv5l77sdzzn3OPfcxq6mpqan5p5yUXkoryo5xYLc0J7RnSVNJ/P9rbJM+SW+ldUXfsCCgi4Vtu/RFWlzY/zg54BlpdWvX0Fgq7SpspPPPwjY2nLPWdIan0q3CNhbks1ryQdpfGqtYJH2XGun3akuvwwPPS9elE9IL6Ye0MPWTVuiNtCrZG8G+Q3ouPTB/Bmv1U2ROS++SCI5nPgr9C8L/vthg7vym1L5s7mDpzD5zRzNzzXc8B7xRumHNgJm/XHptvt5jaWsaS6X9KG1J7W6wSa+kI+bpfFtaI92TjoVxfYNTOHMl2HLVK4vDJfOxbFDejGXWeqYOWTPgzE3zeRScDJuEjTW7wYbwRjNPrOlT9puCNRDscLnTOM1i/EZ2Sl9T32fpjPlRiPQKOGcCVAU839pfxDNpZWj3mt8R3gwpMlnY75ovNlHYM6fM5zEGbQ59vQKOVAVM+lJ9c1bwRuNYjh9neW+wVcJdOSMdLuzvrd1BOCDtCW3OI+czfggMK2DsDWtmRXn/XrDW/r7IZ3J9sK01L0ydCgLjSf8IOxzPUa+AB0lpil08ajwj1wqq/Lf0fyBy6nLNUHGpfix23NorNOQNYhxwzqZDmyp9zbzgnTV3EMdJTeYdNP+4ZzOOJtsdaQmTO0DAzMWXKXMfCZyqTfUeGM4A6Tvb3JGq9KBgcOfNM3e6m6PDho8Krs0q/yphhxulcQThrVIkfwveFgGXBWsUuW/t3wQDwwKkcyxYo8pD635F1tT8j/wCduKOoeLWn7AAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAbCAYAAACa9mScAAABIElEQVR4XmNgGAWDCjgDsQS6IC6gC8SaaGLGQPwVShMF1gBxNJoYiP8fiDnRxHGCg0AsiCZ2Goivo4nhBXPQBYDgExC3ogviAtJArI8uCAQ7GIj0ijIQFwOxBRCzIonzA3EEEh8DMAPxDyDeyACxaSsQMwKxKxC/BmJ5hFLsAGTbdCBugLJBXjmDJA+yHRQ+LEhiGKCMATXaXID4GEKaQQaInwOxEpIYBgAZcACJj54+TBkIJDAOBoghyNF2ngFVQzoQXwViESQxDAAypByJj5w+woH4HwMkZvCCnwyQ+AcBUODBvAIKm/dAfAXKxwtAUekExH8ZINH5jgFiuzlUjiTAA8SdQCzOgJrISALo6YMsYMMAybkUgSIgnoQuOMIAAAgXLSB/on02AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAaCAYAAABhJqYYAAAAtUlEQVR4XmNgGAUjHKgB8Skg3gDEr4DYCVUaAVyA+D0QL4byhYH4GRCrwFVAAScQ/wNiDzTx/0C8FE2MwRKIrwKxCJIYCwNE8R4kMQZeID4MxNHIgkBgygBRnI4sqAnEb4FYH1kQCHKA+AkQKyILTmKAmCCIJBbOAPEDP5IYGBxggCiOBGJGKAaFymkkNXAAUgjyHAcQSwKxOKo0KgApXoMuiA3IAPFvILZBl8AGQIrQw3cwAwBwhR4ZCZCoVAAAAABJRU5ErkJggg==>