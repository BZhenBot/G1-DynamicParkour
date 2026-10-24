# 发行证据索引

- FILE_DECISIONS.csv：每项Git拟公开文件状态、SHA、类别、许可与风险。
- SOURCE_EXCLUSIONS.csv：源Git当前文件入选/排除及大类排除说明。
- scan-summary.json：原/新Git全部对象、拟公开文件与权重扫描及边界。
- TEST_RESULTS.md、test-results.json：独立环境测试，区分通过、失败、未执行。
- dependency-licenses.json：新环境已安装依赖版本/许可元数据，不带二进制。
- upstream-sources.json：上游许可证URL/抓取SHA。
- release内各manifest：权重角色、依赖、整体SHA和发布判断。
- results：历史验收证据去身份副本；原proof SHA不等于公开文件SHA。
- 本地_local_review：安装/运行原始stdout/stderr、扫描命中位置（无密钥明文）、原工作区保护对比。已被Git忽略，不应上传。

无RL episode的审计记不适用；真实回放独立保存配置、种子、源码、权重和事件，不对未执行项填零。
