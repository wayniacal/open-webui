<script>
	import { createEventDispatcher, getContext, onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	import {
		getMemoryProposals,
		applyMemoryProposals,
		discardMemoryProposals
	} from '$lib/apis/memories';

	import Modal from '$lib/components/common/Modal.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import XMark from '$lib/components/icons/XMark.svelte';

	const dispatch = createEventDispatcher();
	const i18n = getContext('i18n');

	export let show;

	let loading = false;
	let saving = false;
	// Working copy. `keep` starts true so the default action is to accept what
	// was proposed — the point of the review is to catch the ones that are
	// wrong, not to re-enter the ones that are right.
	let entries = [];

	const load = async () => {
		loading = true;
		const proposals = await getMemoryProposals(localStorage.token).catch((error) => {
			toast.error(`${error}`);
			return null;
		});

		entries = (proposals ?? [])
			// `remove` proposals have no content to review and are the review
			// asking to forget something; they are shown but not editable.
			.map((proposal) => ({
				id: proposal.id,
				action: proposal.action ?? 'add',
				memoryId: proposal.memory_id ?? null,
				content: proposal.content ?? '',
				type: proposal.type ?? 'context',
				path: proposal.path ?? '',
				keep: true,
				added: false
			}));
		loading = false;
	};

	$: if (show) {
		load();
	}

	const addOwn = () => {
		entries = [
			...entries,
			{
				id: `new-${entries.length}`,
				action: 'add',
				memoryId: null,
				content: '',
				type: 'context',
				path: '',
				keep: true,
				added: true
			}
		];
	};

	const saveHandler = async () => {
		saving = true;

		// What is sent is the outcome, not a set of approvals: rejected entries
		// are simply absent, edits travel as edited, and anything the user typed
		// themselves is indistinguishable from a proposal once stored.
		const memories = entries
			.filter((entry) => entry.keep && (entry.action !== 'add' || entry.content.trim()))
			.map((entry) => ({
				action: entry.action,
				id: entry.memoryId ?? undefined,
				content: entry.content.trim() || undefined,
				type: entry.type,
				path: entry.path.trim() || undefined
			}));

		const res = await applyMemoryProposals(localStorage.token, memories).catch((error) => {
			toast.error(`${error}`);
			return null;
		});

		if (res) {
			toast.success(
				$i18n.t('{{stored}} of {{total}} memories saved', {
					stored: memories.length,
					total: entries.length
				})
			);
			show = false;
			dispatch('save');
		}

		saving = false;
	};

	const discardHandler = async () => {
		saving = true;
		const res = await discardMemoryProposals(localStorage.token).catch((error) => {
			toast.error(`${error}`);
			return null;
		});
		if (res) {
			toast.success($i18n.t('Discarded all suggested memories'));
			show = false;
			dispatch('save');
		}
		saving = false;
	};
</script>

<Modal bind:show size="md">
	<div>
		<div class=" flex justify-between dark:text-gray-300 px-5 pt-4 pb-2">
			<div class=" text-lg font-medium self-center">
				{$i18n.t('Review Memories')}
			</div>
			<button class="self-center" on:click={() => (show = false)}>
				<XMark className={'size-5'} />
			</button>
		</div>

		<div class="flex flex-col w-full px-5 pb-4 dark:text-gray-200">
			<div class="text-xs text-gray-500 mb-3">
				{$i18n.t(
					'These were suggested from your conversations and have not been saved. Uncheck anything you do not want, edit the wording, or add your own.'
				)}
			</div>

			{#if loading}
				<div class="flex justify-center py-6"><Spinner /></div>
			{:else if entries.length === 0}
				<div class="text-sm text-gray-500 py-6 text-center">
					{$i18n.t('Nothing awaiting review')}
				</div>
			{:else}
				<div class="flex flex-col gap-3 max-h-96 overflow-y-auto pr-1">
					{#each entries as entry (entry.id)}
						<div
							class="flex gap-2.5 items-start rounded-lg px-3 py-2.5 bg-gray-50 dark:bg-gray-850 {entry.keep
								? ''
								: 'opacity-40'}"
						>
							<input
								type="checkbox"
								class="mt-1 shrink-0"
								bind:checked={entry.keep}
								aria-label={$i18n.t('Keep this memory')}
							/>

							<div class="flex flex-col w-full min-w-0">
								{#if entry.action === 'add' || entry.action === 'replace'}
									<textarea
										bind:value={entry.content}
										disabled={!entry.keep}
										class="bg-transparent w-full text-sm outline-hidden resize-y"
										rows="2"
										placeholder={$i18n.t('Add a preference, fact, or instruction about you')}
									/>
								{:else}
									<div class="text-sm">
										{$i18n.t('Forget an existing memory')}
										{#if entry.content}
											<span class="opacity-60">— {entry.content}</span>
										{/if}
									</div>
								{/if}

								<div class="flex items-center gap-3 mt-1">
									<button
										type="button"
										class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
										disabled={!entry.keep}
										on:click={() => {
											entry.type = entry.type === 'user' ? 'context' : 'user';
										}}
									>
										{entry.type === 'user' ? $i18n.t('User') : $i18n.t('Context')}
									</button>

									<input
										bind:value={entry.path}
										disabled={!entry.keep}
										class="text-xs bg-transparent outline-hidden text-gray-500 w-full"
										placeholder={$i18n.t('Path')}
										autocomplete="off"
									/>

									{#if entry.added}
										<span class="text-xs text-gray-400 shrink-0">{$i18n.t('Yours')}</span>
									{/if}
								</div>
							</div>
						</div>
					{/each}
				</div>
			{/if}

			<!-- Disabled while loading: `load()` replaces `entries` wholesale when
			     it resolves, so a row added before then would be silently thrown
			     away — the one thing this modal must never do to something the
			     user typed. -->
			<button
				type="button"
				class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 self-start mt-3 disabled:opacity-40"
				disabled={loading}
				on:click={addOwn}
			>
				+ {$i18n.t('Add something it missed')}
			</button>

			<div class="flex justify-between items-center pt-4 text-sm font-medium">
				<button
					class="px-3.5 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition"
					type="button"
					disabled={saving || loading}
					on:click={discardHandler}
				>
					{$i18n.t('Discard all')}
				</button>

				<button
					class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full flex items-center gap-2 whitespace-nowrap {saving
						? ' cursor-not-allowed'
						: ''}"
					type="button"
					disabled={saving || loading}
					on:click={saveHandler}
				>
					{$i18n.t('Save')}

					{#if saving}
						<span class="shrink-0">
							<Spinner />
						</span>
					{/if}
				</button>
			</div>
		</div>
	</div>
</Modal>
