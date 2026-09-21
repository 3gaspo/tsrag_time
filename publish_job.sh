#!/bin/bash
# Manually publish selected TIME logs and artifacts from the DGX Git owner.
set -euo pipefail

usage() {
    echo "usage: bash publish_job.sh [JOB_ID] [--size lightweight|detailed|full] [--clean] [--message TEXT] [--project-root PATH]" >&2
}

project_root="$(pwd)"
job_id=""
message=""
publish_size="lightweight"
clean_mode=false
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    job_id="$1"
    shift
fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        --job-id) job_id="$2"; shift 2 ;;
        --size) publish_size="$2"; shift 2 ;;
        --clean) clean_mode=true; shift ;;
        --message) message="$2"; shift 2 ;;
        --project-root) project_root="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$publish_size" in
    lightweight|detailed|full) ;;
    *) usage; echo "publication size must be lightweight, detailed, or full" >&2; exit 2 ;;
esac
if [ -n "$job_id" ] && ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    usage
    echo "JOB_ID must be numeric" >&2
    exit 2
fi
if [ "$clean_mode" = true ] && { [ -n "$job_id" ] || [ "$publish_size" != lightweight ]; }; then
    usage
    echo "--clean cannot be combined with JOB_ID or --size" >&2
    exit 2
fi

project_root="$(cd "$project_root" && pwd)"
cd "$project_root"
[ "$(git rev-parse --show-toplevel)" = "$project_root" ] || {
    echo "run from a project Git root or pass --project-root: $project_root" >&2
    exit 1
}
[ "$(git symbolic-ref --short HEAD)" = main ] || {
    echo "publisher requires the main branch" >&2
    exit 1
}

proxy_script="${PROXY_SCRIPT_PATH:-$HOME/codes/proxy.sh}"
[ -f "$proxy_script" ] || {
    echo "proxy script not found: $proxy_script" >&2
    exit 1
}
. "$proxy_script"
git pull --ff-only origin main

if [ "$clean_mode" = true ]; then
    deleted_paths=()
    while IFS= read -r -d '' deleted_path; do
        deleted_paths+=("$deleted_path")
    done < <(git diff --name-only --diff-filter=D -z HEAD --)
    stage_paths=()
    clean_paths=()
    for deleted_path in "${deleted_paths[@]}"; do
        clean_paths+=(":(literal)$deleted_path")
        if git ls-files --error-unmatch -- ":(literal)$deleted_path" >/dev/null 2>&1; then
            stage_paths+=(":(literal)$deleted_path")
        fi
    done
    for environment_file in pyproject.toml uv.lock; do
        if [ -f "$environment_file" ]; then
            clean_paths+=(":(literal)$environment_file")
            stage_paths+=(":(literal)$environment_file")
        fi
    done

    if [ "${#clean_paths[@]}" -gt 0 ]; then
        clean_pathspec="$(mktemp)"
        stage_pathspec="$(mktemp)"
        trap 'rm -f -- "$clean_pathspec" "$stage_pathspec"' EXIT
        printf '%s\0' "${clean_paths[@]}" > "$clean_pathspec"
        if [ "${#stage_paths[@]}" -gt 0 ]; then
            printf '%s\0' "${stage_paths[@]}" > "$stage_pathspec"
            git add -v -f -A --pathspec-from-file="$stage_pathspec" --pathspec-file-nul
        fi
        if [ "${#deleted_paths[@]}" -gt 0 ] || ! git diff --cached --quiet -- pyproject.toml uv.lock; then
            [ -n "$message" ] || message="maintenance: publish deletions and environment files"
            git commit --only -m "$message" --pathspec-from-file="$clean_pathspec" --pathspec-file-nul
        else
            echo "No clean-mode changes; pushing existing local commits."
        fi
    else
        echo "No clean-mode changes; pushing existing local commits."
    fi
    git push origin main
    exit 0
fi

paths=()
if [ -n "$job_id" ]; then
    shopt -s nullglob
    out_logs=(
        "$project_root"/logs/*_"$job_id".out
        "$project_root"/logs/*_"$job_id"_*.out
        "$project_root"/logs/selena/*_"$job_id".out
        "$project_root"/logs/selena/*_"$job_id"_*.out
    )
    shopt -u nullglob
    [ "${#out_logs[@]}" -gt 0 ] || {
        echo "no local or synchronized Selena logs found for job $job_id" >&2
        exit 1
    }
    for out_log in "${out_logs[@]}"; do
        err_log="${out_log%.out}.err"
        [ -f "$err_log" ] || {
            echo "missing stderr pair for $out_log" >&2
            exit 1
        }
        paths+=(
            "${out_log#"$project_root"/}"
            "${err_log#"$project_root"/}"
        )
    done
    for status_root in logs/workflow_status logs/selena/workflow_status; do
        [ -d "$status_root" ] || continue
        while IFS= read -r -d '' status_file; do
            if grep -qxF "slurm_job_id=$job_id" "$status_file"; then
                paths+=("$status_file")
            fi
        done < <(find "$status_root" -type f -name '*.status' -print0)
    done
    for metadata_root in logs/dataset_metadata logs/selena/dataset_metadata; do
        if [ -d "$metadata_root/$job_id" ]; then
            paths+=("$metadata_root/$job_id")
        fi
    done
    [ -n "$message" ] || message="slurm: publish job $job_id"
else
    [ -d logs ] || {
        echo "logs directory not found" >&2
        exit 1
    }
    paths+=(logs)
    [ -n "$message" ] || message="slurm: publish $publish_size logs and outputs"
fi

if [ "$publish_size" = full ]; then
    [ ! -d outputs ] || paths+=(outputs)
elif [ -d outputs ]; then
    selection_file="$(mktemp)"
    python3 "$project_root/src/timebench/pipeline/artifact_selection.py" paths "$project_root/outputs" \
        --size "$publish_size" --max-bytes "${PUBLISH_MAX_FILE_BYTES:-100000000}" > "$selection_file"
    while IFS= read -r -d '' artifact; do
        paths+=("${artifact#"$project_root"/}")
    done < "$selection_file"
    rm -f -- "$selection_file"
fi

exclusions=(
    ':(exclude,glob)**/*.pt'
    ':(exclude,glob)**/*.npy'
    ':(exclude,glob)**/*.cbm'
)
max_publish_bytes="${PUBLISH_MAX_FILE_BYTES:-100000000}"
max_sample_bytes="${PUBLISH_SAMPLE_MAX_BYTES:-10000000}"
for limit in "$max_publish_bytes" "$max_sample_bytes"; do
    [[ "$limit" =~ ^[1-9][0-9]*$ ]] || {
        echo "publisher byte limits must be positive integers" >&2
        exit 2
    }
done
[ "$max_sample_bytes" -lt "$max_publish_bytes" ] || {
    echo "PUBLISH_SAMPLE_MAX_BYTES must be smaller than PUBLISH_MAX_FILE_BYTES" >&2
    exit 2
}

sample_paths=()
oversize_exclusions=()
oversize_paths=()
for selected_path in "${paths[@]}"; do
    while IFS= read -r -d '' file; do
        relative="${file#"$project_root"/}"
        case "$relative" in
            *.pt|*.npy|*.cbm) continue ;;
        esac
        file_bytes="$(stat -c '%s' -- "$file")"
        [ "$file_bytes" -gt "$max_publish_bytes" ] || continue

        sample_relative="${relative}.sample.txt"
        sample_file="$project_root/$sample_relative"
        stale_at_utc=""
        if [ -f "$sample_file" ]; then
            stale_at_utc="$(sed -n 's/^git_stale_at_utc: //p' "$sample_file" | head -n 1)"
        fi
        [ -n "$stale_at_utc" ] || stale_at_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        sample_bytes=$(( (file_bytes + 9) / 10 ))
        if [ "$sample_bytes" -gt "$max_sample_bytes" ]; then
            sample_bytes="$max_sample_bytes"
        fi
        mkdir -p -- "$(dirname "$sample_file")"
        {
            echo "Oversized publication artifact sample"
            echo "source: $relative"
            echo "original_bytes: $file_bytes"
            echo "git_stale_at_utc: $stale_at_utc"
            echo "git_stale_reason: associated file became stale on Git due to file size"
            if LC_ALL=C grep -Iq -m 1 . -- "$file"; then
                echo "sample: first $sample_bytes bytes (10% capped at $max_sample_bytes bytes)"
                echo
                head -c "$sample_bytes" -- "$file"
            else
                echo "sample: binary or empty content omitted"
            fi
        } > "$sample_file"
        sample_paths+=("$sample_relative")
        oversize_paths+=("$relative")
        oversize_exclusions+=(":(exclude,literal)$relative")
        echo "Replacing oversized artifact ($file_bytes bytes) with $sample_relative"
    done < <(find "$project_root/$selected_path" -type f -print0)
done
declare -A oversize_lookup=()
for oversize_path in "${oversize_paths[@]}"; do
    oversize_lookup["$oversize_path"]=1
done
publish_paths=()
for selected_path in "${paths[@]}"; do
    [ -z "${oversize_lookup[$selected_path]+x}" ] || continue
    publish_paths+=("$selected_path")
done
publish_paths+=("${sample_paths[@]}")
publish_pathspec="$(mktemp)"
trap 'rm -f -- "$publish_pathspec"' EXIT
printf '%s\0' "${publish_paths[@]}" "${exclusions[@]}" "${oversize_exclusions[@]}" > "$publish_pathspec"

if [ -n "$job_id" ]; then
    echo "Publishing job $job_id logs and $publish_size TIME artifacts:"
else
    echo "Publishing DGX and synchronized Selena logs plus $publish_size TIME artifacts:"
fi
printf '  %s\n' "${paths[@]}"
git add -v -f --pathspec-from-file="$publish_pathspec" --pathspec-file-nul
if ! git diff --cached --quiet; then
    git commit --only -m "$message" --pathspec-from-file="$publish_pathspec" --pathspec-file-nul
else
    echo "No new artifact changes; pushing existing local commits."
fi
git push origin main
