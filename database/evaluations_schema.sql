create table if not exists evaluations (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null constraint evaluations_owner_fk references "user"(id) on delete restrict,
  run_id uuid not null references training_tasks(id) on delete cascade,
  dataset_version_id uuid not null references datasets(id) on delete restrict,
  compute_job_id uuid unique references compute_jobs(id) on delete set null,
  origin text not null check (origin in ('automatic','manual')),
  status text not null default 'awaiting_queue' check (status in
    ('awaiting_queue','queued','running','stopping','completed','partial','no_test','failed','cancelled')),
  execution jsonb not null default '{"mode":"auto","gpuUuid":null}'::jsonb,
  dataset_content_sha256 text,
  checkpoint_sha256 text,
  result_path text,
  error_message text,
  idempotency_key text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  finished_at timestamptz
);
do $$ begin
  if not exists (select 1 from pg_constraint where conname='evaluations_owner_fk' and conrelid='evaluations'::regclass) then
    alter table evaluations add constraint evaluations_owner_fk foreign key(owner_user_id) references "user"(id) on delete restrict;
  end if;
end $$;
create unique index if not exists evaluations_automatic_run on evaluations(run_id) where origin='automatic';
create unique index if not exists evaluations_owner_key on evaluations(owner_user_id,idempotency_key) where idempotency_key is not null;
create index if not exists evaluations_owner_run on evaluations(owner_user_id,run_id,created_at desc);
create index if not exists evaluations_pending on evaluations(created_at) where status='awaiting_queue';
create index if not exists evaluations_dataset_status on evaluations(dataset_version_id,status);

create table if not exists evaluation_images (
  evaluation_id uuid not null references evaluations(id) on delete cascade,
  ordinal integer not null,
  filename text not null,
  source_path text not null,
  outcome text not null,
  score double precision,
  record_offset bigint not null,
  record_length integer not null,
  primary key(evaluation_id,ordinal)
);
alter table evaluation_images add column if not exists fp_count integer not null default 0;
alter table evaluation_images add column if not exists fn_count integer not null default 0;
create index if not exists evaluation_images_outcome on evaluation_images(evaluation_id,outcome,ordinal);
create index if not exists evaluation_images_score on evaluation_images(evaluation_id,score,ordinal);

create or replace function guard_active_compute_run() returns trigger language plpgsql as $$
begin
  if exists(select 1 from compute_jobs where run_id=old.id and status in ('queued','dispatching','running','stopping','recovery_pending'))
     or exists(select 1 from evaluations where run_id=old.id and status in ('awaiting_queue','queued','running','stopping')) then
    raise exception 'Cancel active compute jobs before deleting or transferring this run.' using errcode='55006';
  end if;
  return old;
end $$;
