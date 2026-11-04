create table if not exists dataset_indexes (
  dataset_version_id uuid primary key references datasets(id) on delete cascade,
  owner_user_id text not null references "user"(id) on delete cascade,
  status text not null default 'queued' check(status in ('queued','running','completed','failed')),
  active_generation uuid,
  working_generation uuid not null,
  rq_job_id text,
  total integer not null default 0,
  processed integer not null default 0,
  skipped integer not null default 0,
  stats jsonb not null default '{}'::jsonb,
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists dataset_indexes_owner on dataset_indexes(owner_user_id,status);

create table if not exists dataset_index_generations (
  dataset_version_id uuid not null references datasets(id) on delete cascade,
  generation uuid not null,
  cache_created boolean not null default false,
  created_at timestamptz not null default now(),
  primary key(dataset_version_id,generation)
);

create table if not exists dataset_index_images (
  dataset_version_id uuid not null references datasets(id) on delete cascade,
  generation uuid not null,
  ordinal integer not null,
  split text not null check(split in ('train','val','test')),
  filename text not null,
  source_path text not null,
  thumbnail_path text not null,
  mask_path text,
  width integer not null,
  height integer not null,
  pixel_sha256 text not null,
  class_names text[] not null default '{}',
  has_objects boolean not null default false,
  annotations jsonb not null default '{}'::jsonb,
  primary key(dataset_version_id,generation,ordinal)
);
create index if not exists dataset_index_images_split on dataset_index_images(dataset_version_id,generation,split,ordinal);
create index if not exists dataset_index_images_hash on dataset_index_images(dataset_version_id,generation,pixel_sha256,split);
create index if not exists dataset_index_images_classes on dataset_index_images using gin(class_names);
