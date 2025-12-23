import click
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs

@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass

@cli.command()
def download_models():
    """Download required docling models."""
    click.echo("Downloading docling models...")
    Pipeline.download_docling_models()

@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel worker processes')
def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing."""
    # === 修改点 1: 不再手动获取 cwd，让 Pipeline 内部自动定位 ===
    pipeline = Pipeline()
    
    click.echo(f"Parsing PDFs (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

@cli.command()
@click.option('--max-workers', default=10, help='Number of workers for table serialization')
def serialize_tables(max_workers):
    """Serialize tables in parsed reports using parallel threading."""
    # === 修改点 2: 实例化时不传路径 ===
    pipeline = Pipeline()
    
    click.echo(f"Serializing tables (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='Configuration preset to use')
def process_reports(config):
    """Process parsed reports through the pipeline stages."""
    run_config = preprocess_configs[config]
    # === 修改点 3: 只传 run_config ===
    pipeline = Pipeline(run_config=run_config)
    
    click.echo(f"Processing parsed reports (config={config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking']), default='max_nst_o3m', help='Configuration preset to use')
def process_questions(config):
    """Process questions using the pipeline."""
    # 增加校验，防止配置名写错
    if config not in configs:
         click.echo(f"Error: Config '{config}' not found in pipeline.py.")
         return

    run_config = configs[config]
    # === 修改点 4: 只传 run_config ===
    pipeline = Pipeline(run_config=run_config)
    
    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()