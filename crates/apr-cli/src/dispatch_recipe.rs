// PMAT-5230: `apr merge|finetune|distill`, with and without `--recipe`. Kept
// out of `dispatch_model_commands` so the recipe arms do not raise its
// complexity; each function handles one command, both ways.

fn not_this_command(name: &str) -> CliError {
    CliError::ValidationFailed(format!("internal: {name} dispatch got another command"))
}

/// The `--set key=value` values as plain strings.
fn texts(set: &[FreeText]) -> Vec<String> {
    set.iter().map(ToString::to_string).collect()
}

fn dispatch_merge(cli: &Cli, cmd: &Commands) -> Result<(), CliError> {
    use crate::commands::train_recipe::RecipeKind;
    use crate::commands::train_recipe_run::{self, wrong_plan, Planned};
    match cmd {
        Commands::Merge {
            base_model,
            drop_rate,
            density,
            seed,
            plan,
            force,
            recipe: Some(recipe),
            set,
            ..
        } => train_recipe_run::run(recipe, &texts(set), RecipeKind::Merge, *plan, |p| {
            let Planned::Merge(a) = p else {
                return Err(wrong_plan(p));
            };
            let r = a
                .models
                .iter()
                .map(|m| crate::error::resolve_model_path(Path::new(m)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            merge::run(
                &r,
                a.strategy,
                Some(Path::new(&a.output)),
                a.weights.clone(),
                base_model.as_deref().map(Path::to_path_buf),
                *drop_rate,
                *density,
                *seed,
                cli.json,
                *plan,
                *force,
            )
        }),
        Commands::Merge {
            files,
            strategy,
            output,
            weights,
            base_model,
            drop_rate,
            density,
            seed,
            plan,
            force,
            recipe: None,
            set,
        } => {
            if let Err(e) = train_recipe_run::sets_need_recipe(&texts(set)) {
                return Err(e);
            }
            let resolved: std::result::Result<Vec<std::path::PathBuf>, _> = files
                .iter()
                .map(|f| crate::error::resolve_model_path(f))
                .collect();
            match resolved {
                Ok(r) => merge::run(
                    &r,
                    strategy,
                    output.as_deref(),
                    weights.clone(),
                    base_model.as_deref().map(Path::to_path_buf),
                    *drop_rate,
                    *density,
                    *seed,
                    cli.json,
                    *plan,
                    *force,
                ),
                Err(e) => Err(e),
            }
        }
        _ => Err(not_this_command("merge")),
    }
}

#[cfg(feature = "training")]
fn dispatch_finetune(cli: &Cli, cmd: &ModelOpsCommands) -> Result<(), CliError> {
    use crate::commands::train_recipe::RecipeKind;
    use crate::commands::train_recipe_run::{self, wrong_plan, Planned};
    match cmd {
        ModelOpsCommands::Finetune {
            file,
            method,
            rank,
            vram,
            plan,
            data,
            output,
            adapter,
            merge,
            epochs,
            learning_rate,
            model_size,
            task,
            num_classes,
            checkpoint_format,
            oversample,
            max_seq_len,
            quantize_nf4,
            gpus,
            gpu_backend,
            role,
            bind,
            coordinator,
            expect_workers,
            wait_gpu,
            adapters,
            adapters_config,
            experimental_mps,
            gpu_share,
            profile,
            recipe,
            set,
        } => {
            if *profile {
                eprintln!("StepProfiler enabled for finetune (PMAT-486)");
            }
            // The flags a recipe covers; the rest pass through unchanged.
            let call = |file: Option<&Path>,
                        method: &str,
                        rank: Option<u32>,
                        data: Option<&Path>,
                        output: Option<&Path>,
                        epochs: u32,
                        learning_rate: Option<f64>,
                        max_seq_len: Option<usize>| {
                finetune::run(
                    file,
                    method,
                    rank,
                    *vram,
                    *plan,
                    data,
                    output,
                    adapter.as_deref(),
                    *merge,
                    epochs,
                    learning_rate,
                    model_size.as_deref(),
                    task.as_deref(),
                    *num_classes,
                    checkpoint_format,
                    *oversample,
                    max_seq_len,
                    *quantize_nf4,
                    gpus.as_deref(),
                    gpu_backend,
                    role.as_deref(),
                    bind.as_deref(),
                    coordinator.as_deref(),
                    *expect_workers,
                    *wait_gpu,
                    &batuta_common::cli_roles::strings(adapters),
                    adapters_config.as_deref(),
                    cli.json,
                    *experimental_mps,
                    *gpu_share,
                )
            };
            match recipe {
                Some(recipe) => {
                    train_recipe_run::run(recipe, &texts(set), RecipeKind::Finetune, *plan, |p| {
                        let Planned::Finetune(a) = p else {
                            return Err(wrong_plan(p));
                        };
                        call(
                            Some(Path::new(&a.model)),
                            a.method,
                            a.rank,
                            Some(Path::new(&a.data)),
                            Some(Path::new(&a.output)),
                            a.epochs,
                            Some(a.learning_rate),
                            Some(a.max_seq_len),
                        )
                    })
                }
                None => {
                    if let Err(e) = train_recipe_run::sets_need_recipe(&texts(set)) {
                        return Err(e);
                    }
                    call(
                    file.as_deref(),
                    method,
                    *rank,
                    data.as_deref(),
                    output.as_deref(),
                    *epochs,
                    *learning_rate,
                    *max_seq_len,
                    )
                }
            }
        }
        _ => Err(not_this_command("finetune")),
    }
}

fn dispatch_distill(cli: &Cli, cmd: &ModelOpsCommands) -> Result<(), CliError> {
    use crate::commands::train_recipe::RecipeKind;
    use crate::commands::train_recipe_run::{self, wrong_plan, Planned};
    match cmd {
        ModelOpsCommands::Distill {
            strategy,
            plan,
            stage,
            backend,
            dataset,
            recipe: Some(recipe),
            set,
            ..
        } => train_recipe_run::run(recipe, &texts(set), RecipeKind::Distill, *plan, |p| {
            let Planned::Distill(a) = p else {
                return Err(wrong_plan(p));
            };
            distill::run(
                Some(Path::new(&a.teacher)),
                Some(Path::new(&a.student)),
                Some(Path::new(&a.data)),
                Some(Path::new(&a.output)),
                strategy,
                a.temperature,
                a.alpha,
                a.epochs,
                *plan,
                None,
                stage.as_deref(),
                backend.as_str(),
                dataset.as_deref(),
                cli.json,
            )
        }),
        ModelOpsCommands::Distill {
            teacher,
            student,
            data,
            output,
            strategy,
            temperature,
            alpha,
            epochs,
            plan,
            config,
            stage,
            backend,
            dataset,
            recipe: None,
            set,
        } => {
            if let Err(e) = train_recipe_run::sets_need_recipe(&texts(set)) {
                return Err(e);
            }
            distill::run(
            teacher.as_deref(),
            student.as_deref(),
            data.as_deref(),
            output.as_deref(),
            strategy,
            *temperature,
            *alpha,
            *epochs,
            *plan,
            config.as_deref(),
            stage.as_deref(),
            backend.as_str(),
            dataset.as_deref(),
            cli.json,
            )
        }
        _ => Err(not_this_command("distill")),
    }
}
