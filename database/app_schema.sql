create extension if not exists pgcrypto;

create table if not exists workspaces (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  slug text not null unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists workspace_members (
  workspace_id uuid not null references workspaces(id) on delete cascade,
  user_id text not null,
  role text not null check (role in ('owner', 'admin', 'member', 'viewer')),
  created_at timestamptz not null default now(),
  primary key (workspace_id, user_id)
);

create table if not exists projects (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  name text not null,
  slug text not null,
  task_type text,
  created_by text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (workspace_id, slug)
);

create table if not exists datasets (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  project_id uuid references projects(id) on delete set null,
  name text not null,
  slug text not null,
  storage_path text not null,
  formats jsonb not null default '[]'::jsonb,
  tasks jsonb not null default '[]'::jsonb,
  classes jsonb not null default '[]'::jsonb,
  created_by text not null,
  created_at timestamptz not null default now(),
  unique (workspace_id, slug)
);

alter table datasets alter column workspace_id drop not null;
alter table datasets add column if not exists owner_user_id text;
alter table datasets add column if not exists status text not null default 'active';
alter table datasets add column if not exists metadata jsonb not null default '{}'::jsonb;
alter table datasets add column if not exists updated_at timestamptz not null default now();

do $$
begin
  if to_regclass('training_tasks') is null and to_regclass('training_runs') is not null then
    alter table training_runs rename to training_tasks;
  end if;
end $$;

create table if not exists training_tasks (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  project_id uuid references projects(id) on delete set null,
  dataset_id uuid references datasets(id) on delete set null,
  rq_job_id text unique,
  run_slug text,
  display_name text not null default 'cv_run',
  task_type text not null,
  model_type text not null,
  model_name text,
  params jsonb not null default '{}'::jsonb,
  status text not null default 'queued',
  storage_path text,
  latest_metrics jsonb,
  created_by text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  finished_at timestamptz,
  unique (workspace_id, run_slug)
);

alter table training_tasks alter column workspace_id drop not null;
alter table training_tasks alter column run_slug drop not null;
alter table training_tasks alter column storage_path drop not null;
alter table training_tasks add column if not exists display_name text;
alter table training_tasks add column if not exists owner_user_id text;
alter table training_tasks add column if not exists dataset_slug text;
alter table training_tasks add column if not exists error_detail text;
update training_tasks
set display_name = regexp_replace(run_slug, '_[0-9]{13}$', '')
where display_name is null and run_slug is not null;
update training_tasks set display_name = 'cv_run' where display_name is null;
alter table training_tasks alter column display_name set not null;
alter table training_tasks alter column display_name set default 'cv_run';

create table if not exists artifacts (
  id uuid primary key default gen_random_uuid(),
  training_run_id uuid not null references training_tasks(id) on delete cascade,
  kind text not null,
  file_name text not null,
  storage_path text not null,
  content_type text,
  size_bytes bigint,
  created_at timestamptz not null default now()
);

do $$
begin
  if exists (select 1 from pg_constraint where conname = 'training_runs_owner_user_fk')
     and not exists (select 1 from pg_constraint where conname = 'training_tasks_owner_user_fk') then
    alter table training_tasks rename constraint training_runs_owner_user_fk to training_tasks_owner_user_fk;
  end if;
  if to_regclass('idx_training_runs_workspace_id') is not null and to_regclass('idx_training_tasks_workspace_id') is null then
    alter index idx_training_runs_workspace_id rename to idx_training_tasks_workspace_id;
  end if;
  if to_regclass('idx_training_runs_created_by') is not null and to_regclass('idx_training_tasks_created_by') is null then
    alter index idx_training_runs_created_by rename to idx_training_tasks_created_by;
  end if;
  if to_regclass('idx_runs_owner_slug') is not null and to_regclass('idx_tasks_owner_slug') is null then
    alter index idx_runs_owner_slug rename to idx_tasks_owner_slug;
  end if;
  if to_regclass('idx_runs_dataset_status') is not null and to_regclass('idx_tasks_dataset_status') is null then
    alter index idx_runs_dataset_status rename to idx_tasks_dataset_status;
  end if;
end $$;

create index if not exists idx_workspace_members_user_id on workspace_members(user_id);
create index if not exists idx_projects_workspace_id on projects(workspace_id);
create index if not exists idx_datasets_workspace_id on datasets(workspace_id);
create index if not exists idx_training_tasks_workspace_id on training_tasks(workspace_id);
create index if not exists idx_training_tasks_created_by on training_tasks(created_by);
create unique index if not exists idx_datasets_owner_slug on datasets(owner_user_id, slug) where owner_user_id is not null;
create unique index if not exists idx_tasks_owner_slug on training_tasks(owner_user_id, run_slug) where owner_user_id is not null;
create index if not exists idx_tasks_dataset_status on training_tasks(dataset_id, status);

create table if not exists annotation_projects (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null,
  name text not null,
  task_type text not null check (task_type in ('image_classification', 'object_detection', 'semantic_segmentation', 'instance_segmentation')),
  classes jsonb not null default '[]'::jsonb,
  train_ratio integer not null default 80 check (train_ratio between 1 and 100),
  val_ratio integer not null default 10 check (val_ratio between 0 and 99),
  test_ratio integer not null default 10 check (test_ratio between 0 and 99),
  split_seed text not null default gen_random_uuid()::text,
  status text not null default 'draft' check (status in ('draft', 'publishing', 'published')),
  published_version integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (train_ratio + val_ratio + test_ratio = 100)
);

-- Upgrade an existing annotation table in place without deleting legacy data.
alter table annotation_projects
  add column if not exists train_ratio integer not null default 80 check (train_ratio between 1 and 100),
  add column if not exists val_ratio integer not null default 10 check (val_ratio between 0 and 99),
  add column if not exists test_ratio integer not null default 10 check (test_ratio between 0 and 99),
  add column if not exists split_seed text not null default gen_random_uuid()::text;

do $migration$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'annotation_projects'::regclass
      and conname = 'annotation_projects_status_check'
  ) then
    update annotation_projects set status = 'draft'
    where status not in ('draft', 'publishing', 'published');
    alter table annotation_projects add constraint annotation_projects_status_check
      check (status in ('draft', 'publishing', 'published'));
  end if;
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'annotation_projects'::regclass
      and conname = 'annotation_projects_split_total_check'
  ) then
    alter table annotation_projects add constraint annotation_projects_split_total_check
      check (train_ratio + val_ratio + test_ratio = 100);
  end if;
end $migration$;

alter table annotation_projects alter column status set default 'draft';

create table if not exists annotation_images (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references annotation_projects(id) on delete cascade,
  file_name text not null,
  storage_path text not null,
  width integer not null check (width > 0),
  height integer not null check (height > 0),
  sort_order integer not null,
  split text check (split in ('train', 'val', 'test')),
  annotations jsonb not null default '[]'::jsonb,
  marked_empty boolean not null default false,
  is_labeled boolean not null default false,
  revision integer not null default 0,
  updated_at timestamptz not null default now(),
  unique (project_id, file_name)
);

alter table annotation_images
  add column if not exists is_excluded boolean not null default false,
  add column if not exists split_source text not null default 'manual',
  add column if not exists content_sha256 text,
  add column if not exists thumbnail_path text,
  add column if not exists thumbnail_width integer,
  add column if not exists thumbnail_height integer;

do $migration$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'annotation_images'::regclass
      and conname = 'annotation_images_split_source_check'
  ) then
    alter table annotation_images add constraint annotation_images_split_source_check
      check (split_source in ('auto', 'manual'));
  end if;
end $migration$;

create table if not exists annotation_operations (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references annotation_projects(id) on delete cascade,
  owner_user_id text not null,
  kind text not null check (kind in ('import', 'publish')),
  status text not null default 'queued' check (status in ('queued', 'running', 'completed', 'failed', 'cancelled')),
  progress integer not null default 0 check (progress between 0 and 100),
  processed_count integer not null default 0,
  total_count integer not null default 0,
  skipped_count integer not null default 0,
  upload_path text,
  result jsonb not null default '{}'::jsonb,
  error_detail text,
  rq_job_id text,
  attempt integer not null default 0,
  cancel_requested boolean not null default false,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz,
  updated_at timestamptz not null default now()
);

create table if not exists annotation_image_revisions (
  id uuid primary key default gen_random_uuid(),
  image_id uuid not null references annotation_images(id) on delete cascade,
  revision integer not null,
  annotations jsonb not null default '[]'::jsonb,
  marked_empty boolean not null default false,
  is_excluded boolean not null default false,
  created_at timestamptz not null default now(),
  unique (image_id, revision)
);

create index if not exists idx_annotation_projects_owner on annotation_projects(owner_user_id, updated_at desc);
create index if not exists idx_annotation_images_project on annotation_images(project_id, sort_order);
create unique index if not exists idx_annotation_images_content
  on annotation_images(project_id, content_sha256) where content_sha256 is not null;
create index if not exists idx_annotation_images_gallery
  on annotation_images(project_id, split, is_excluded, is_labeled, sort_order);
create index if not exists idx_annotation_operations_owner
  on annotation_operations(owner_user_id, created_at desc);
create index if not exists idx_annotation_operations_status
  on annotation_operations(status, updated_at);
create index if not exists idx_annotation_revisions_image
  on annotation_image_revisions(image_id, created_at desc);

do $$
begin
  if to_regclass('"user"') is not null and not exists (select 1 from pg_constraint where conname = 'datasets_owner_user_fk') then
    alter table datasets add constraint datasets_owner_user_fk foreign key (owner_user_id)
      references "user"(id) on delete restrict not valid;
  end if;
  if to_regclass('"user"') is not null and not exists (select 1 from pg_constraint where conname = 'training_tasks_owner_user_fk') then
    alter table training_tasks add constraint training_tasks_owner_user_fk foreign key (owner_user_id)
      references "user"(id) on delete restrict not valid;
  end if;
  if to_regclass('"user"') is not null and not exists (select 1 from pg_constraint where conname = 'annotation_projects_owner_user_fk') then
    alter table annotation_projects add constraint annotation_projects_owner_user_fk foreign key (owner_user_id)
      references "user"(id) on delete restrict not valid;
  end if;
  if to_regclass('"user"') is not null and not exists (select 1 from pg_constraint where conname = 'annotation_operations_owner_user_fk') then
    alter table annotation_operations add constraint annotation_operations_owner_user_fk foreign key (owner_user_id)
      references "user"(id) on delete restrict not valid;
  end if;
end $$;

do $$
begin
  if exists (select 1 from pg_constraint where conname = 'datasets_owner_user_fk' and not convalidated) then
    alter table datasets validate constraint datasets_owner_user_fk;
  end if;
  if exists (select 1 from pg_constraint where conname = 'training_tasks_owner_user_fk' and not convalidated) then
    alter table training_tasks validate constraint training_tasks_owner_user_fk;
  end if;
  if exists (select 1 from pg_constraint where conname = 'annotation_projects_owner_user_fk' and not convalidated) then
    alter table annotation_projects validate constraint annotation_projects_owner_user_fk;
  end if;
  if exists (select 1 from pg_constraint where conname = 'annotation_operations_owner_user_fk' and not convalidated) then
    alter table annotation_operations validate constraint annotation_operations_owner_user_fk;
  end if;
end $$;
