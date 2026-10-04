create table if not exists label_studio_accounts (
  owner_user_id text primary key references "user"(id) on delete restrict,
  ls_user_id bigint not null unique,
  created_at timestamptz not null default now()
);
create table if not exists label_studio_projects (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null references label_studio_accounts(owner_user_id) on delete restrict,
  ls_project_id bigint not null unique,
  name text not null,
  kind text not null check (kind in ('image_classification','object_detection','semantic_segmentation','instance_segmentation')),
  published_version integer not null default 0,
  deleted_at timestamptz,
  created_at timestamptz not null default now()
);
create index if not exists label_studio_projects_owner on label_studio_projects(owner_user_id);
alter table label_studio_projects add column if not exists published_revision bigint;
alter table label_studio_projects add column if not exists reserved_version integer not null default 0;
create table if not exists label_studio_operations (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references label_studio_projects(id) on delete restrict,
  owner_user_id text not null references label_studio_accounts(owner_user_id) on delete restrict,
  kind text not null check (kind in ('import','publish')),
  status text not null default 'queued' check (status in ('queued','running','completed','failed','cancelled')),
  processed integer not null default 0,
  total integer not null default 0,
  skipped integer not null default 0,
  upload_path text,
  snapshot_id text,
  version integer,
  result jsonb not null default '{}',
  error text,
  cancel_requested boolean not null default false,
  attempt integer not null default 0,
  rq_job_id text,
  heartbeat_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create unique index if not exists label_studio_one_active_operation
  on label_studio_operations(project_id) where status in ('queued','running');
create index if not exists label_studio_operations_owner on label_studio_operations(owner_user_id,created_at desc);
alter table label_studio_operations add column if not exists preview_only boolean not null default false;
alter table label_studio_operations add column if not exists error_code text;
alter table label_studio_operations add column if not exists error_details jsonb not null default '{}';
update label_studio_projects p set reserved_version=greatest(p.reserved_version,p.published_version,
  coalesce((select max(o.version) from label_studio_operations o where o.project_id=p.id),0));
