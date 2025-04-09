import importlib
import inspect
import sys # sysモジュールをインポート

"""
DEFORUM STORAGE IMPORT
deforum_constants.py から DeforumStorage クラスをインポートします。
シングルトンパターンで実装されており、Deforumノード間で状態を共有するために使用されます。
"""
# storageモジュールへのパスが正しいか確認が必要な場合があります
# 環境によっては .modules.deforum_constants から ..modules.deforum_constants のように変更
try:
    # mapping.py から見た相対パスでインポートを試みます
    from .modules.deforum_constants import DeforumStorage
except ImportError:
    # 上記が失敗した場合、一つ上の階層からインポートを試みます（代替）
    print("[deforum] Warning: Could not import DeforumStorage from .modules, trying ..modules")
    try:
        from ..modules.deforum_constants import DeforumStorage
    except ImportError as e:
        print(f"[deforum] Error: Failed to import DeforumStorage. Ensure 'deforum_constants.py' exists in the correct 'modules' directory relative to 'mapping.py'. Error details: {e}")
        # エラーが発生した場合でも処理を続けるために、ダミークラスやNoneを設定することも検討できます
        # 例: DeforumStorage = None
        raise  # または、ここでエラーを再発生させて起動を停止させる

# DeforumStorage のインスタンスを取得または生成
gs = DeforumStorage() if DeforumStorage is not None else None

"""
NODE CLASS IMPORTS
各カテゴリのDeforumカスタムノードクラスをインポートします。
ファイルが存在しない場合やインポートエラーが発生した場合に備え、
特にオプションのノードについては try-except で囲みます。
"""
print("[deforum] Importing node classes...")
try:
    from .nodes.deforum_audiosync_nodes import *
    print("[deforum] Imported audiosync nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import audiosync nodes: {e}")

try:
    from .nodes.deforum_cache_nodes import *
    print("[deforum] Imported cache nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import cache nodes: {e}")

try:
    from .nodes.deforum_cnet_nodes import *
    print("[deforum] Imported cnet nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import cnet nodes: {e}")

try:
    from .nodes.deforum_cond_nodes import *
    print("[deforum] Imported cond nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import cond nodes: {e}")

try:
    from .nodes.deforum_data_nodes import *
    print("[deforum] Imported data nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import data nodes: {e}")

try:
    from .nodes.deforum_framewarp_node import *
    print("[deforum] Imported framewarp node.")
except ImportError as e:
    print(f"[deforum] Failed to import framewarp node: {e}")

try:
    from .nodes.deforum_hybrid_nodes import *
    print("[deforum] Imported hybrid nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import hybrid nodes: {e}")

try:
    from .nodes.deforum_interpolation_nodes import *
    print("[deforum] Imported interpolation nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import interpolation nodes: {e}")

try:
    from .nodes.deforum_image_nodes import *
    print("[deforum] Imported image nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import image nodes: {e}")

try:
    from .nodes.deforum_iteration_nodes import *
    print("[deforum] Imported iteration nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import iteration nodes: {e}")

try:
    from .nodes.deforum_legacy_nodes import *
    print("[deforum] Imported legacy nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import legacy nodes: {e}")

try:
    from .nodes.deforum_logic_nodes import *
    print("[deforum] Imported logic nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import logic nodes: {e}")

# オプションのノイズノード
try:
    from .nodes.deforum_noise_nodes import AddCustomNoiseNode
    print("[deforum] Imported noise node.")
except ImportError:
    print("[deforum] Optional node AddCustomNoiseNode not found or failed to import.")
    AddCustomNoiseNode = None # インポート失敗時にNoneを設定

# オプションの高度なノイズノード
try:
    from .nodes.deforum_advnoise_node import AddAdvancedNoiseNode
    print("[deforum] Imported advanced noise node.")
except ImportError:
    print("[deforum] Optional node AddAdvancedNoiseNode not found or failed to import.")
    AddAdvancedNoiseNode = None # インポート失敗時にNoneを設定

try:
    from .nodes.deforum_prompt_nodes import *
    print("[deforum] Imported prompt nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import prompt nodes: {e}")

try:
    from .nodes.redirect_console_node import DeforumRedirectConsole
    print("[deforum] Imported redirect console node.")
except ImportError as e:
    print(f"[deforum] Failed to import redirect console node: {e}")

try:
    from .nodes.deforum_sampler_nodes import *
    print("[deforum] Imported sampler nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import sampler nodes: {e}")

try:
    from .nodes.deforum_schedule_visualizer import *
    print("[deforum] Imported schedule visualizer.")
except ImportError as e:
    print(f"[deforum] Failed to import schedule visualizer: {e}")

try:
    from .nodes.deforum_video_nodes import *
    print("[deforum] Imported video nodes.")
except ImportError as e:
    print(f"[deforum] Failed to import video nodes: {e}")

# exec_hijack モジュールをインポート (存在する場合)
# これはComfyUIの実行フローに介入するためのモジュールかもしれません
try:
    from . import exec_hijack
    print("[deforum] Imported exec_hijack module.")
except ImportError:
    print("[deforum] exec_hijack module not found or failed to import.")
    exec_hijack = None # インポート失敗時にNoneを設定

"""
NODE MAPPING
ComfyUIがカスタムノードを認識できるように、クラス名とクラスオブジェクト、
およびクラス名と表示名をマッピングします。
"""
# NODE_CLASS_MAPPINGS と NODE_DISPLAY_NAME_MAPPINGS を初期化
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("[deforum] Starting node mapping...")

# 現在のモジュールオブジェクトを取得
# __name__ はこのファイル (mapping.py) のモジュール名を指します
current_module = sys.modules[__name__]

# 現在のモジュール内で定義またはインポートされたすべてのメンバーを反復処理
for name, obj in inspect.getmembers(current_module):
    # メンバーがクラスであり、かつ INPUT_TYPES 属性を持っているか確認
    # これはComfyUIカスタムノードクラスの基本的な規約です
    if inspect.isclass(obj) and hasattr(obj, "INPUT_TYPES"):
        # クラス名を取得 (例: 'DeforumLoadVideoNode')
        class_name = name
        # 表示名を取得
        # クラスに CATEGORY 属性があればそれを使用 (例: 'deforum/video')
        category = getattr(obj, 'CATEGORY', 'deforum') # カテゴリがない場合は 'deforum' をデフォルトに
        # クラスに display_name 属性があればそれを使用、なければクラス名を整形
        # (例: 'Load Video (Deforum)' のような表示名)
        node_display_name = getattr(obj, "display_name", class_name.replace("Deforum", "").replace("Node", ""))
        full_display_name = f"{node_display_name} (Deforum)"

        # マッピングに追加
        NODE_CLASS_MAPPINGS[class_name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[class_name] = full_display_name # キーをクラス名に統一

        # print(f"[deforum] Mapped Class: {class_name} -> {full_display_name}") # 詳細ログが必要な場合

print("[deforum] Deforum custom nodes mapping complete.")
# 登録されたノードの数を表示
print(f"[deforum] Total mapped classes: {len(NODE_CLASS_MAPPINGS)}")
# 必要であれば、登録されたクラス名の一覧を表示
# print(f"[deforum] Mapped Classes: {list(NODE_CLASS_MAPPINGS.keys())}")

