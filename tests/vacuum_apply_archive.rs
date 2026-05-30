//! AC3: `--apply --action archive` moves the file and removes the row.
//!
//! After apply:
//! - file is at `memories-archive/<kind>/<id>.md`
//! - original path is gone
//! - `vacuum_candidates` no longer returns the id (row deleted)

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use recall::vacuum::{VacuumConfig, apply_archive};

#[test]
fn vacuum_apply_archive_moves_file_and_removes_row() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();
    let vcfg = VacuumConfig::default();

    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "memory to archive");
    mem.front.confidence = 0.55;
    mem.front.surfaced_count = 25;
    let src_path = store.write(&mem).unwrap();
    idx.upsert(&mem, &src_path, None).unwrap();

    assert!(src_path.exists(), "memory file must exist before archive");

    let candidates = idx
        .vacuum_candidates(vcfg.min_surfaced, vcfg.max_used)
        .unwrap();
    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];

    apply_archive(&idx, root, c, &src_path).unwrap();

    // Original file should be gone.
    assert!(
        !src_path.exists(),
        "original file should be gone after archive"
    );

    // Archive destination should exist.
    let dst = root
        .join("memories-archive")
        .join("semantic")
        .join(format!("{}.md", mem.front.id));
    assert!(dst.exists(), "archive destination should exist at {}", dst.display());

    // Row should be removed from the index.
    let after = idx
        .vacuum_candidates(1, u32::MAX)
        .unwrap();
    assert!(
        after.iter().all(|candidate| candidate.id != mem.front.id),
        "archived memory must not appear in vacuum_candidates after archive"
    );
}
