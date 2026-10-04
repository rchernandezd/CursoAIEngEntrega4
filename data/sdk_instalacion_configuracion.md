# techcorp-sdk — Instalación y configuración

`techcorp-sdk` es la librería interna de Python que mantiene el Equipo Plataforma para
consumir la API de Recursos Humanos de TechCorp. Esta página cubre la instalación del
paquete, la verificación de la instalación y todas las formas de configurarlo. La
documentación corresponde a la versión 3.4 del paquete; el módulo que se importa en el
código se llama `tcsdk`.

## Requisitos

La librería requiere Python 3.10 o superior, porque utiliza anotaciones de tipos y
coincidencia de patrones que no existen en versiones anteriores del intérprete. Funciona
en Windows, Linux y macOS. Se recomienda trabajar siempre dentro de un entorno virtual
por proyecto, para no mezclar dependencias con otras aplicaciones ni con el Python del
sistema operativo.

## Instalación

El paquete se publica en el índice interno de paquetes de la empresa, que ya está
configurado en los equipos corporativos. Para instalarlo dentro de un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\Activate.ps1
pip install techcorp-sdk==3.4.*
```

Fijar la versión mayor y menor con `==3.4.*` permite recibir correcciones sin cambios
incompatibles. Para actualizar a una versión nueva, revisar primero las notas de la
versión en el portal del Equipo Plataforma.

## Verificar la instalación

Una vez instalado, se puede comprobar que el módulo se importa correctamente y que la
versión es la esperada con un solo comando:

```bash
python -c "import tcsdk; print(tcsdk.__version__)"
```

El comando debe imprimir un número que comience con 3.4. Si aparece un error de módulo
no encontrado, lo más habitual es que el entorno virtual no esté activado en la terminal
actual. Si el número de versión es distinto, es probable que exista otra instalación
del paquete en el Python global: conviene desinstalarla y repetir la instalación con el
entorno virtual activado. En integración continua se recomienda agregar este mismo
comando como primer paso del pipeline, para detectar a tiempo una versión inesperada.

## Token de acceso

Cada aplicación que usa la librería necesita un token personal o de servicio, que se
genera desde el portal del Equipo Plataforma. El token se configura solo a través de la
variable de entorno `TCSDK_API_TOKEN`. Nunca debe escribirse en el código fuente, en
archivos de configuración versionados ni subirse al repositorio: un token expuesto debe
revocarse de inmediato desde el mismo portal.

```python
import os

import tcsdk

# El token se toma del entorno; jamás se escribe en el código.
if "TCSDK_API_TOKEN" not in os.environ:
    raise SystemExit("Definí la variable de entorno TCSDK_API_TOKEN antes de ejecutar.")

config = tcsdk.load_config()
print(config.env, config.timeout)
```

## Archivo tcsdk.toml y entornos

Además de las variables de entorno, la librería admite un archivo opcional `tcsdk.toml`
en la raíz del proyecto, útil para valores que no son secretos, como el entorno o el
timeout. Cuando un mismo valor aparece en varios lugares, la precedencia es: argumentos
pasados en el código > variables de entorno > `tcsdk.toml`.

La variable `TCSDK_ENV` indica contra qué entorno de la API se trabaja y acepta los
valores `dev`, `staging` o `prod`. Si no se define, el valor por defecto es `dev`, de
modo que ninguna prueba local impacte por accidente sobre datos productivos.

```toml
# tcsdk.toml
env = "staging"
timeout = 45
```

## Timeout y otros parámetros

Cada solicitud a la API tiene un timeout por defecto de 30 segundos. Puede ajustarse en
`tcsdk.toml`, con la variable de entorno correspondiente o directamente en el código
al cargar la configuración:

```python
import tcsdk

# Los argumentos tienen la máxima precedencia sobre entorno y archivo.
config = tcsdk.load_config(env="prod", timeout=10)
assert config.timeout == 10
```

Valores de timeout demasiado bajos generan errores intermitentes en consultas grandes;
se recomienda no bajar de 10 segundos salvo en procesos con requisitos de latencia muy
estrictos.
