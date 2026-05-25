//! Filesystem-backed memory store. Markdown files are the source of truth.

use crate::memory::Memory;
use crate::paths;
use anyhow::{Context, Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    pub fn open(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(paths::memories_dir(&root))
            .with_context(|| format!("create memories dir under {}", root.display()))?;
        fs::create_dir_all(paths::index_dir(&root))?;
        fs::create_dir_all(paths::session_dir(&root))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn write(&self, mem: &Memory) -> Result<PathBuf> {
        let path = paths::memory_path(&self.root, &mem.front.subject, &mem.front.id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let md = mem.to_markdown()?;
        write_atomic(&path, &md)?;
        Ok(path)
    }

    /// Overwrite an existing memory at `path` with the new contents, atomically.
    /// Used by `recall update` so the id and existing file location are preserved.
    #[allow(clippy::unused_self)]
    pub fn overwrite(&self, path: &Path, mem: &Memory) -> Result<()> {
        let md = mem.to_markdown()?;
        write_atomic(path, &md)?;
        Ok(())
    }

    #[allow(clippy::unused_self)]
    pub fn read(&self, path: &Path) -> Result<Memory> {
        let text =
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Memory::from_markdown(&text).with_context(|| format!("parse {}", path.display()))
    }

    #[allow(clippy::unused_self)]
    pub fn delete(&self, path: &Path) -> Result<()> {
        fs::remove_file(path).with_context(|| format!("remove {}", path.display()))
    }

    /// Find a memory by id. Walks `memories/` — fine for the few-thousand-file
    /// scale we expect; if it ever isn't, the `SQLite` index already tracks
    /// `path` per id.
    pub fn find_by_id(&self, id: &str) -> Result<(Memory, PathBuf)> {
        for entry in WalkDir::new(paths::memories_dir(&self.root))
            .into_iter()
            .filter_map(Result::ok)
        {
            if entry.file_type().is_file()
                && entry.file_name() == format!("{id}.md").as_str()
            {
                let mem = self.read(entry.path())?;
                return Ok((mem, entry.path().to_path_buf()));
            }
        }
        Err(anyhow!("memory id {id} not found"))
    }

    /// Iterate every memory under `memories/` (used by `reindex`).
    pub fn iter_all(&self) -> impl Iterator<Item = Result<(Memory, PathBuf)>> + '_ {
        WalkDir::new(paths::memories_dir(&self.root))
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_type().is_file()
                    && e.path().extension().is_some_and(|x| x == "md")
            })
            .map(move |e| {
                let mem = self.read(e.path())?;
                Ok((mem, e.path().to_path_buf()))
            })
    }

    /// Walk every `*.md` under `memories/` and return `(id, path)` pairs
    /// without parsing frontmatter. Used by `doctor` to detect orphans cheaply.
    pub fn iter_ids(&self) -> impl Iterator<Item = (String, PathBuf)> + '_ {
        WalkDir::new(paths::memories_dir(&self.root))
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_type().is_file()
                    && e.path().extension().is_some_and(|x| x == "md")
            })
            .filter_map(|e| {
                let stem = e.path().file_stem()?.to_str()?.to_string();
                Some((stem, e.path().to_path_buf()))
            })
    }
}

/// Write `contents` to `path` atomically: write to a sibling temp file then
/// rename. On POSIX, rename is atomic within the same filesystem, so readers
/// either see the old contents or the full new contents — never a partial write.
fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("path {} has no parent", path.display()))?;
    fs::create_dir_all(dir)?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("path {} has no file name", path.display()))?
        .to_string_lossy()
        .to_string();
    let tmp = dir.join(format!(".{file_name}.tmp"));
    fs::write(&tmp, contents).with_context(|| format!("write tempfile {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memory::{Kind, Subject};

    #[test]
    fn write_then_read_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileStore::open(tmp.path().to_path_buf()).unwrap();
        let mem = Memory::new(Kind::Semantic, Subject::user(), "test body");
        let path = store.write(&mem).unwrap();
        assert!(path.exists());
        let back = store.read(&path).unwrap();
        assert_eq!(back.front.id, mem.front.id);
        assert_eq!(back.body.trim(), "test body");
    }

    #[test]
    fn find_by_id_walks_subdirs() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileStore::open(tmp.path().to_path_buf()).unwrap();
        let mem = Memory::new(Kind::Procedural, Subject::project("recall"), "build with cargo");
        store.write(&mem).unwrap();
        let (found, _) = store.find_by_id(&mem.front.id).unwrap();
        assert_eq!(found.front.id, mem.front.id);
    }

    #[test]
    fn iter_all_returns_every_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileStore::open(tmp.path().to_path_buf()).unwrap();
        for i in 0..5 {
            let m = Memory::new(Kind::Semantic, Subject::user(), format!("m{i}"));
            store.write(&m).unwrap();
        }
        let count = store.iter_all().filter_map(Result::ok).count();
        assert_eq!(count, 5);
    }
}
