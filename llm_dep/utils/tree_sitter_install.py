from tree_sitter import Language

Language.build_library(
  # so文件保存位置
  'deprecated_cllm/utils/tree_sitter.so',

  # vendor文件下git clone的仓库
  [
    # 'vendor/tree-sitter-java',
    'deprecated_cllm/utils/tree-sitter-python-master',
    # 'vendor/tree-sitter-cpp',
    # 'vendor/tree-sitter-c-sharp',
    # 'vendor/tree-sitter-javascript',
  ]
)
