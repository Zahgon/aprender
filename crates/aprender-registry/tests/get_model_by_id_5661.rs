//! `RegistryDb::get_model_by_id` fetches exactly the row with that id, round-trips
//! every field, and reports a missing id as `NotFound` (#5661).

use pacha::model::{Model, ModelCard, ModelId, ModelStage, ModelVersion};
use pacha::registry::RegistryDb;
use pacha::storage::ContentAddress;
use pacha::PachaError;

fn model(name: &str, major: u32) -> Model {
    Model {
        id: ModelId::new(),
        name: name.to_string(),
        version: ModelVersion::new(major, 0, 0),
        content_address: ContentAddress::from_bytes(name.as_bytes()),
        card: ModelCard::new(format!("{name} card")),
        stage: ModelStage::Development,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    }
}

fn db() -> (tempfile::TempDir, RegistryDb) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = RegistryDb::open(dir.path().join("r.db")).expect("open");
    (dir, db)
}

#[test]
fn round_trips_the_inserted_model() {
    let (_d, db) = db();
    let m = model("fraud", 1);
    db.insert_model(&m).expect("insert");
    let got = db.get_model_by_id(&m.id).expect("get");
    assert_eq!(got.id, m.id);
    assert_eq!(got.name, "fraud");
    assert_eq!(got.version, m.version);
    assert_eq!(got.content_address, m.content_address);
    assert_eq!(got.stage, ModelStage::Development);
    assert_eq!(got.card.description, "fraud card");
}

#[test]
fn selects_by_id_among_several_rows() {
    let (_d, db) = db();
    let (a, b) = (model("a", 1), model("b", 2));
    db.insert_model(&a).expect("insert a");
    db.insert_model(&b).expect("insert b");
    assert_eq!(db.get_model_by_id(&b.id).expect("b").name, "b");
    assert_eq!(db.get_model_by_id(&a.id).expect("a").name, "a");
}

#[test]
fn unknown_id_is_not_found_naming_the_id() {
    let (_d, db) = db();
    db.insert_model(&model("a", 1)).expect("insert");
    let missing = ModelId::new();
    match db.get_model_by_id(&missing) {
        Err(PachaError::NotFound { kind, name, .. }) => {
            assert_eq!(kind, "model");
            assert_eq!(name, missing.to_string());
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn stage_update_is_visible_through_the_id_lookup() {
    let (_d, db) = db();
    let m = model("s", 1);
    db.insert_model(&m).expect("insert");
    db.update_model_stage(&m.id, ModelStage::Production).expect("stage");
    assert_eq!(db.get_model_by_id(&m.id).expect("get").stage, ModelStage::Production);
}
