import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

console = Console()


# ── Pydantic model ───────────────────────────────────────────────
class JobPosting(BaseModel):
    title: str = Field(description="Job title exactly as intended, cleaned up")
    company: str = Field(description="Company name")
    salary_min: Optional[int] = Field(None, description="Minimum salary as integer, null if not mentioned")
    salary_max: Optional[int] = Field(None, description="Maximum salary as integer, null if not mentioned")
    currency: Optional[str] = Field(None, description="Currency code e.g. USD, PKR, GBP, null if unknown")
    skills: list[str] = Field(description="Required technical skills as a clean list")
    location: str = Field(description="City and country, or Remote if fully remote")
    remote: bool = Field(description="True if remote work is allowed or mentioned")
    experience_years: Optional[int] = Field(None, description="Minimum years of experience required, null if not mentioned")
    seniority: Optional[str] = Field(None, description="Seniority level: Junior, Mid, Senior, Lead, or null")

    @field_validator("skills")
    @classmethod
    def clean_skills(cls, v: list[str]) -> list[str]:
        # Remove duplicates, strip whitespace
        return list(dict.fromkeys(s.strip() for s in v if s.strip()))

    @field_validator("salary_max")
    @classmethod
    def max_gte_min(cls, v: Optional[int], info) -> Optional[int]:
        if v is not None and "salary_min" in info.data:
            if info.data["salary_min"] is not None and v < info.data["salary_min"]:
                raise ValueError("salary_max must be >= salary_min")
        return v


# ── 10 test job postings ─────────────────────────────────────────
JOB_POSTINGS = [
    {
        "id": 1,
        "label": "Clean posting",
        "text": """
        Senior Python Engineer at DataFlow Inc.
        Location: San Francisco, CA (Remote OK)
        Salary: $140,000 - $180,000 USD
        Requirements: 5+ years Python, FastAPI, PostgreSQL, Docker, AWS
        """
    },
    {
        "id": 2,
        "label": "Messy startup style",
        "text": """
        we need a rockstar fullstack dev!! must know react + node.js, bonus if u know 
        typescript. we're a startup in london, mostly remote but come in sometimes. 
        paying 60-80k gbp depending on exp. junior to mid level ok
        """
    },
    {
        "id": 3,
        "label": "Missing salary",
        "text": """
        Machine Learning Engineer — Karachi, Pakistan
        Company: TechVentures PK
        Skills needed: Python, TensorFlow, PyTorch, scikit-learn, pandas
        At least 3 years of ML experience required. On-site only.
        Competitive salary offered.
        """
    },
    {
        "id": 4,
        "label": "PKR salary",
        "text": """
        Hiring: Backend Developer at SoftHouse Lahore
        Pay: 150k-250k PKR per month
        Must have: Django, REST APIs, MySQL, Git
        Fresh to 2 years experience. Office-based in Lahore.
        """
    },
    {
        "id": 5,
        "label": "Fully remote",
        "text": """
        REMOTE ONLY - Senior DevOps Engineer
        Anywhere in the world. We are CloudOps Ltd.
        Salary range: $120k-$160k USD annually
        Must know: Kubernetes, Terraform, AWS, CI/CD, Linux
        7+ years in DevOps or infrastructure roles
        """
    },
    {
        "id": 6,
        "label": "Vague skills",
        "text": """
        Looking for a data analyst to join our finance team in Dubai.
        You should be good with numbers and spreadsheets, know some SQL,
        maybe Power BI or Tableau. Salary negotiable. Office role.
        Company: FinEdge Analytics
        """
    },
    {
        "id": 7,
        "label": "Multiple skill formats",
        "text": """
        Frontend Developer needed urgently!
        Skills: React.js, Vue.js, HTML5/CSS3, JavaScript ES6+, TypeScript, 
        Webpack, Git, REST API integration
        Salary: $70,000 to $90,000
        Location: Austin TX, hybrid (3 days office)
        Company: UIcraft
        """
    },
    {
        "id": 8,
        "label": "Academic/research role",
        "text": """
        Research Engineer - NLP & LLMs
        Anthropic-style research lab, based in London UK
        We're looking for someone with PhD or equivalent, strong Python,
        experience with transformers, HuggingFace, RLHF, fine-tuning.
        Salary: £90,000 - £130,000. Hybrid working.
        """
    },
    {
        "id": 9,
        "label": "No company name",
        "text": """
        Exciting opportunity for a React Native developer!
        Build mobile apps for iOS and Android.
        Remote position, timezone: EST preferred
        $80k-100k USD. Need: React Native, Redux, REST APIs, 2+ years mobile dev
        """
    },
    {
        "id": 10,
        "label": "Non-English mixed",
        "text": """
        Python Developer required for Islamabad office.
        Salary: 200,000 to 300,000 PKR
        Skills: Python, Flask, MongoDB, Docker
        Freshers may also apply. Company: NetSol Technologies.
        Timing: 9am-6pm, Mon-Fri
        """
    },
]


# ── Extractor ────────────────────────────────────────────────────
def extract_job_posting(raw_text: str, llm) -> JobPosting:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a job posting parser. Extract structured information from the job posting. "
         "For salary, always convert to integers (e.g. '150k' → 150000, '£90,000' → 90000). "
         "For remote: true if remote, hybrid, or 'remote ok' is mentioned. "
         "For seniority: infer from title or experience (0-2 yrs=Junior, 2-5=Mid, 5+=Senior). "
         "If a field is not mentioned, use null."),
        ("user", "Extract from this job posting:\n\n{text}")
    ])
    chain = prompt | llm
    return chain.invoke({"text": raw_text})


def display_result(job: JobPosting, posting: dict) -> None:
    salary = "Not mentioned"
    if job.salary_min and job.salary_max:
        currency = job.currency or ""
        salary = f"{currency} {job.salary_min:,} - {job.salary_max:,}"
    elif job.salary_min:
        salary = f"{job.currency or ''} {job.salary_min:,}+"

    console.print(Panel(
        f"[bold]{job.title}[/bold] at [cyan]{job.company}[/cyan]\n"
        f"[dim]Location:[/dim] {job.location} | "
        f"[dim]Remote:[/dim] {'✓' if job.remote else '✗'} | "
        f"[dim]Seniority:[/dim] {job.seniority or 'N/A'} | "
        f"[dim]Experience:[/dim] {str(job.experience_years) + ' yrs' if job.experience_years else 'N/A'}\n"
        f"[dim]Salary:[/dim] {salary}\n"
        f"[dim]Skills:[/dim] {', '.join(job.skills)}",
        title=f"[yellow]#{posting['id']} {posting['label']}[/yellow]",
        border_style="blue"
    ))


def main() -> None:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    ).with_structured_output(JobPosting)

    console.print(Rule("[bold cyan]Lab 2.1 — Structured Output Extractor[/bold cyan]"))
    console.print("[dim]Extracting structured data from 10 messy job postings...[/dim]\n")

    results = []
    errors = []

    for posting in JOB_POSTINGS:
        with console.status(f"[dim]Extracting #{posting['id']}: {posting['label']}...[/dim]", spinner="dots"):
            try:
                job = extract_job_posting(posting["text"], llm)
                results.append((posting, job))
                display_result(job, posting)
            except Exception as e:
                errors.append((posting["id"], str(e)))
                console.print(f"[red]#{posting['id']} failed: {e}[/red]")

    # ── Summary table ─────────────────────────────────────────────
    console.print(Rule("[bold cyan]Extraction Summary[/bold cyan]"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#",        width=3)
    table.add_column("Label",    width=20)
    table.add_column("Title",    width=25)
    table.add_column("Salary",   width=20)
    table.add_column("Remote",   width=7)
    table.add_column("Skills",   width=30)

    for posting, job in results:
        salary = "—"
        if job.salary_min and job.salary_max:
            salary = f"{job.currency or ''} {job.salary_min//1000}k-{job.salary_max//1000}k"

        table.add_row(
            str(posting["id"]),
            posting["label"],
            job.title[:23],
            salary,
            "✓" if job.remote else "✗",
            ", ".join(job.skills[:3]) + ("..." if len(job.skills) > 3 else ""),
        )

    console.print(table)
    console.print(f"\n[green]✓ Extracted {len(results)}/10 successfully[/green]")
    if errors:
        console.print(f"[red]✗ {len(errors)} failed: {errors}[/red]")


if __name__ == "__main__":
    main()
