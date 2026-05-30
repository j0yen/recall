//! AC4: `--apply --action supersede` writes a proposal file.
//!
//! After apply:
//! - `~/.claude/recall/proposals/<ULID>.md` exists
//! - contains target_id, surfaced/used counts, `pending: true`
//! - original memory is unchanged

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use recall::vacuum::{VacuumConfig, apply_supersede_proposal};

#[test]
fn vacuum_apply_supersede_writes_proposal_unchanged_memory() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();
    let vcfg = VacuumConfig::default();

    let body = "supersede proposal test: this memory appeared many times but was never used";
    let mut mem = Memory::new(Kind::Semantic, Subject::user(), body);
    mem.front.confidence = 0.60;
    mem.front.surfaced_count = 22;
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();

    let candidates = idx
        .vacuum_candidates(vcfg.min_surfaced, vcfg.max_used)
        .unwrap();
    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];
    let target_id = c.id.clone();

    let proposal_id = apply_supersede_proposal(root, c, body).unwrap();

    // Proposal file must exist.
    let proposal_path = root.join("proposals").join(format!("{proposal_id}.md"));
    assert!(
        proposal_path.exists(),
        "proposal file should exist at {}",
        proposal_path.display()
    );

    let content = std::fs::read_to_string(&proposal_path).unwrap();

    // Must contain target_id.
    assert!(
        content.contains(&target_id),
        "proposal must reference target_id {target_id}"
    );
    // Must contain pending: true.
    assert!(
        content.contains("pending: true"),
        "proposal must have pending: true"
    );
    // Must contain surfaced count.
    assert!(
        content.contains("surfaced: 22"),
        "proposal must have surfaced: 22"
    );
    // Must contain proposal_type.
    assert!(
        content.contains("proposal_type: vacuum"),
        "proposal must have proposal_type: vacuum"
    );

    // Original memory must be unchanged (confidence still 0.60).
    let candidates_after = idx
        .vacuum_candidates(vcfg.min_surfaced, vcfg.max_used)
        .unwrap();
    let mem_after = candidates_after
        .iter()
        .find(|c| c.id == target_id)
        .expect("original memory must still be in index after supersede");
    assert!(
        (mem_after.confidence_before - 0.60).abs() < 1e-9,
        "original memory confidence must be unchanged; got {:.3}",
        mem_after.confidence_before
    );
}
