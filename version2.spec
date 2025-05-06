# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

a = Analysis(
    ['version2.py'],
    pathex=[],
    binaries=[],
    datas=[
        (os.path.abspath('logo.png'), 'logo.png'),
        (os.path.abspath('user.png'), 'user.png'),
        (os.path.abspath('logo.ico'), 'logo.ico'),
        ('.venv/Lib/site-packages/en_core_web_lg', 'en_core_web_lg'),
        ('C:/Users/SUQOON/AppData/Roaming/nltk_data', 'nltk_data'),
        ('.venv/Lib/site-packages/transformers', 'transformers'),
        ('.venv/Lib/site-packages/torch', 'torch'),
    ],
    hiddenimports=[
        'spacy',
        'en_core_web_lg',
        'numpy',
        'pandas',
        'PyPDF2',
        'tkinter',
        'customtkinter',
        'PIL',
        'io',
        'os',
        'sys',
        'threading',
        'queue',
        'time',
        're',
        'json',
        'datetime',
        'traceback',
        'logging',
        'spacy.lang.en',
        'spacy.tokens',
        'spacy.vocab',
        'spacy.language',
        'thinc',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PDF_Chatbot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo.ico',
)
